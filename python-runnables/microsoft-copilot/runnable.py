# Microsoft Copilot → Dataiku Govern sync (via Microsoft Graph)
from dataiku.runnables import Runnable


class MyRunnable(Runnable):
    """Sync Microsoft Copilot licensing (and optional usage/audit summaries) into Dataiku Govern.

    It creates/updates:
    - A Govern Project representing the Microsoft 365 tenant
    - One External Model per Copilot SKU
    - One External Model Version per service plan under the SKU
    """

    DEFAULTS = {
        "graph_endpoint": "https://graph.microsoft.com",
        "graph_use_beta": False,
        "COPILOT_ONLY_SERVICEPLAN_NAME": None,
        "include_usage_reports": True,
        "usage_period": "D30",
        "usage_report_function": None,
        "include_audit_logs": False,
        "audit_days": 30,
    }

    def __init__(self, project_key, config, plugin_config):
        self.project_key = project_key
        self.config = config or {}
        self.plugin_config = plugin_config or {}

    def get_progress_target(self):
        return None

    def run(self, progress_callback):
        import os
        import json
        import csv
        import io
        from typing import Optional
        from datetime import datetime, timezone, timedelta
        import requests
        import dataikuapi
        from dataikuapi.govern.blueprint import GovernBlueprintVersionId
        from dataikuapi.govern.artifact_search import (
            GovernArtifactSearchQuery,
            GovernArtifactFilterBlueprints,
            GovernArtifactFilterFieldValue,
        )

        def pick(k: str, *, alt: Optional[str] = None):
            return (
                self.config.get(k)
                or self.plugin_config.get(k)
                or os.environ.get(alt or k)
                or self.DEFAULTS.get(alt or k)
            )

        tenant_id = pick("tenant_id")
        client_id = pick("client_id")
        client_secret = pick("client_secret")
        govern_base = pick("govern_base")
        govern_apikey = pick("govern_API_key")
        graph_base = pick("graph_endpoint")
        graph_use_beta = bool(pick("graph_use_beta"))
        only_plan_name = pick("COPILOT_ONLY_SERVICEPLAN_NAME")
        include_usage = bool(pick("include_usage_reports"))
        usage_period = pick("usage_period") or "D30"
        usage_function = pick("usage_report_function")
        include_audit = bool(pick("include_audit_logs"))
        audit_days = int(pick("audit_days") or 30)

        if not all([tenant_id, client_id, client_secret, govern_base, govern_apikey]):
            raise ValueError(
                "Missing required configuration: tenant_id, client_id, client_secret, govern_base, govern_API_key"
            )

        # Auth: Microsoft Graph
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        resp = requests.post(
            token_url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": f"{graph_base}/.default",
                "grant_type": "client_credentials",
            },
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"Graph token error: {resp.status_code} {resp.text}")
        access_token = resp.json().get("access_token")
        if not access_token:
            raise RuntimeError("No access_token from Microsoft identity platform.")

        # Clients
        gc = dataikuapi.GovernClient(govern_base.rstrip("/"), govern_apikey, insecure_tls=True)
        api_ver = "beta" if graph_use_beta else "v1.0"
        headers = {"Authorization": f"Bearer {access_token}"}

        # Consts: Govern blueprint ids and fields
        BP_PARENT_ID, BV_PARENT_ID = "bp.system.govern_project", "bv.iso_42001_v05"
        BP_MODEL_ID, BV_MODEL_ID = "bp.external_models", "bv.azureml_model"
        BP_VER_ID, BV_VER_ID = "bp.external_model_version", "bv.azureml_model_version"
        F_EXT, F_DESC, F_TAGS = "external_id", "description", "tags_json"
        F_VER, F_CRTON, F_CRTBY = "version", "creation_date", "created_by"
        F_LASTMOD, F_MODBY, F_PATH = "last_modified", "modified_by", "path"
        F_PROJ_MODELS, F_MODEL_PARENT, F_VER_PARENT = (
            "govern_models",
            "govern_project",
            "model",
        )

        # Helpers that depend on gc
        def _set_reference(defn, field_id: str, artifact_id: str, list_field: bool = False):
            raw = defn.get_raw()
            fields = raw.setdefault("fields", {})
            cur = fields.get(field_id)
            if list_field:
                if cur is None:
                    fields[field_id] = [artifact_id]
                    return
                if isinstance(cur, list):
                    ids = {(x.get("artifactId") if isinstance(x, dict) else x) for x in cur}
                    if artifact_id not in ids:
                        cur.append({"artifactId": artifact_id} if (cur and isinstance(cur[0], dict)) else artifact_id)
                    fields[field_id] = cur
                    return
                if isinstance(cur, dict) and "artifactId" in cur:
                    fields[field_id] = [cur] if cur.get("artifactId") == artifact_id else [cur, {"artifactId": artifact_id}]
                    return
                fields[field_id] = [cur] if cur == artifact_id else [cur, artifact_id]
                return
            if isinstance(cur, list):
                if artifact_id not in cur:
                    cur.append(artifact_id)
                fields[field_id] = cur
            elif isinstance(cur, dict):
                fields[field_id] = {"artifactId": artifact_id}
            else:
                fields[field_id] = artifact_id

        def find_one_by_field(bp_id: str, field_id: str, value: str):
            q = GovernArtifactSearchQuery()
            q.add_artifact_filter(GovernArtifactFilterBlueprints([bp_id]))
            q.add_artifact_filter(
                GovernArtifactFilterFieldValue("EQUALS", condition=value, field_id=field_id)
            )
            hits = (
                gc.new_artifact_search_request(q)
                .fetch_next_batch(page_size=1)
                .get_response_hits()
            )
            return hits[0].to_artifact() if hits else None

        def upsert_project(tenant_display_name: str):
            name = f"Microsoft 365 Tenant - {tenant_display_name}"
            q = GovernArtifactSearchQuery()
            q.add_artifact_filter(GovernArtifactFilterBlueprints([BP_PARENT_ID]))
            q.add_artifact_filter(GovernArtifactFilterFieldValue("EQUALS", condition=name))
            hits = (
                gc.new_artifact_search_request(q)
                .fetch_next_batch(page_size=1)
                .get_response_hits()
            )
            if hits:
                return hits[0].to_artifact(), "existing"
            payload = {
                "blueprintVersionId": GovernBlueprintVersionId(BP_PARENT_ID, BV_PARENT_ID).build(),
                "name": name,
                "fields": {},
            }
            return gc.create_artifact(payload), "created"

        def upsert_model_shell(meta: dict, project_artifact_id: Optional[str]):
            art = find_one_by_field(BP_MODEL_ID, F_EXT, meta["external_id"])
            fields = {
                F_EXT: meta["external_id"],
                F_DESC: meta.get("description"),
                F_TAGS: json.dumps(meta.get("tags", {})),
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            if art:
                d = art.get_definition()
                raw = d.get_raw()
                raw.setdefault("fields", {}).update(fields)
                if raw.get("name") != meta["name"]:
                    raw["name"] = meta["name"]
                if project_artifact_id:
                    _set_reference(d, F_MODEL_PARENT, project_artifact_id, list_field=False)
                d.save()
                return art, "updated"
            payload = {
                "blueprintVersionId": GovernBlueprintVersionId(BP_MODEL_ID, BV_MODEL_ID).build(),
                "name": meta["name"],
                "fields": fields,
            }
            art = gc.create_artifact(payload)
            if project_artifact_id:
                d = art.get_definition()
                _set_reference(d, F_MODEL_PARENT, project_artifact_id, list_field=False)
                d.save()
            return art, "created"

        def link_model_into_project(project_art, model_artifact_id: str):
            d = project_art.get_definition()
            raw = d.get_raw()
            fields = raw.setdefault("fields", {})
            cur = fields.get(F_PROJ_MODELS)
            if cur is None:
                fields[F_PROJ_MODELS] = [model_artifact_id]
                d.save()
                return
            if isinstance(cur, list):
                ids = {(x.get("artifactId") if isinstance(x, dict) else x) for x in cur}
                if model_artifact_id not in ids:
                    cur.append(
                        {"artifactId": model_artifact_id}
                        if (cur and isinstance(cur[0], dict))
                        else model_artifact_id
                    )
                    fields[F_PROJ_MODELS] = cur
                    d.save()
                    return
            if isinstance(cur, dict) and "artifactId" in cur:
                if cur.get("artifactId") != model_artifact_id:
                    fields[F_PROJ_MODELS] = [cur, {"artifactId": model_artifact_id}]
                    d.save()
                    return
            if isinstance(cur, str) and cur != model_artifact_id:
                fields[F_PROJ_MODELS] = [cur, model_artifact_id]
                d.save()

        def upsert_model_version(meta: dict, model_artifact_id: Optional[str]):
            art = find_one_by_field(BP_VER_ID, F_EXT, meta["external_id"])
            fields = {
                F_EXT: meta["external_id"],
                F_VER: str(meta.get("version")),
                F_DESC: meta.get("description"),
                F_TAGS: json.dumps(meta.get("tags", {})),
                F_CRTON: meta.get("created_on"),
                F_CRTBY: meta.get("created_by"),
                F_LASTMOD: meta.get("last_modified"),
                F_MODBY: meta.get("modified_by"),
                F_PATH: meta.get("path"),
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            title = f"{meta['name']} - v{meta.get('version') or '?'}"
            if art:
                d = art.get_definition()
                raw = d.get_raw()
                raw.setdefault("fields", {}).update(fields)
                if raw.get("name") != title:
                    raw["name"] = title
                if model_artifact_id:
                    _set_reference(d, F_VER_PARENT, model_artifact_id, list_field=False)
                d.save()
                return art, "updated"
            payload = {
                "blueprintVersionId": GovernBlueprintVersionId(BP_VER_ID, BV_VER_ID).build(),
                "name": title,
                "fields": fields,
            }
            art = gc.create_artifact(payload)
            if model_artifact_id:
                d = art.get_definition()
                _set_reference(d, F_VER_PARENT, model_artifact_id, list_field=False)
                d.save()
            return art, "created"

        # Resolve tenant/org
        org = requests.get(f"{graph_base}/{api_ver}/organization", headers=headers, timeout=30)
        org.raise_for_status()
        org_value = (org.json() or {}).get("value", [])
        tenant_display_name = (org_value[0] or {}).get("displayName") if org_value else tenant_id
        project_art, _ = upsert_project(tenant_display_name)

        # Licenses / SKUs
        skus = requests.get(f"{graph_base}/{api_ver}/subscribedSkus", headers=headers, timeout=60)
        skus.raise_for_status()
        sku_items = (skus.json() or {}).get("value", [])

        count_models, count_versions = 0, 0
        for sku in sku_items:
            sku_id = sku.get("skuId")
            sku_part = sku.get("skuPartNumber")
            service_plans = sku.get("servicePlans", []) or []
            if not sku_part or "COPILOT" not in str(sku_part).upper():
                continue

            tags = {
                "capabilityStatus": sku.get("capabilityStatus"),
                "prepaid_units": json.dumps(sku.get("prepaidUnits", {})),
                "consumed_units": sku.get("consumedUnits"),
            }

            # Optional usage report (CSV)
            if include_usage and usage_function:
                try:
                    rep_url = f"{graph_base}/v1.0/reports/{usage_function}(period='{usage_period}')"
                    rep = requests.get(
                        rep_url, headers={**headers, "Accept": "text/csv"}, timeout=60
                    )
                    if rep.ok:
                        reader = csv.DictReader(io.StringIO(rep.text))
                        rows = list(reader)
                        totals = {}
                        for row in rows:
                            for k, v in row.items():
                                if not v:
                                    continue
                                low = k.lower()
                                if low in (
                                    "report refresh date",
                                    "reportdate",
                                    "date",
                                    "lastactivitydate",
                                    "product",
                                ):
                                    continue
                                try:
                                    n = float(str(v).replace(",", ""))
                                except Exception:
                                    continue
                                totals[k] = totals.get(k, 0.0) + n
                        tags[f"usage_{usage_function}"] = json.dumps(
                            {"period": usage_period, "row_count": len(rows), "totals": totals}
                        )
                    else:
                        tags[f"usage_{usage_function}_error"] = str(rep.status_code)
                except Exception as e:
                    tags[f"usage_{usage_function}_error"] = str(e)

            # Optional audit log count (Directory Audits)
            if include_audit:
                try:
                    since = (
                        datetime.now(timezone.utc) - timedelta(days=audit_days)
                    ).isoformat().replace("+00:00", "Z")
                    audit_url = (
                        f"{graph_base}/{api_ver}/auditLogs/directoryAudits?$count=true&$filter=activityDateTime ge {since}"
                    )
                    aud = requests.get(
                        audit_url, headers={**headers, "ConsistencyLevel": "eventual"}, timeout=60
                    )
                    if aud.ok:
                        data = aud.json() or {}
                        count = data.get("@odata.count")
                        if count is None:
                            vals = data.get("value", [])
                            count = len(vals) if isinstance(vals, list) else 0
                        tags["directory_audit_count"] = int(count)
                        tags["directory_audit_lookback_days"] = audit_days
                    else:
                        tags["directory_audit_error"] = str(aud.status_code)
                except Exception as e:
                    tags["directory_audit_error"] = str(e)

            model_meta = {
                "external_id": f"m365:sku:{sku_id}",
                "name": f"Microsoft Copilot - {sku_part}",
                "description": "Microsoft Copilot SKU via Microsoft Graph",
                "tags": tags,
            }
            model_art, _ = upsert_model_shell(
                model_meta, project_artifact_id=project_art.artifact_id
            )
            link_model_into_project(project_art, model_art.artifact_id)
            count_models += 1

            for sp in service_plans:
                sp_name = sp.get("servicePlanName")
                if only_plan_name and sp_name != only_plan_name:
                    continue
                ver = sp.get("servicePlanId")
                state = sp.get("provisioningStatus")
                applies_to = sp.get("appliesTo")
                ver_meta = {
                    "external_id": f"m365:sku:{sku_id}:plan:{ver}",
                    "name": sku_part,
                    "version": ver,
                    "description": f"Service plan {sp_name} (state={state}, appliesTo={applies_to})",
                    "tags": {
                        "servicePlanName": sp_name,
                        "provisioningStatus": state,
                        "appliesTo": applies_to,
                    },
                    "created_on": None,
                    "created_by": None,
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                    "modified_by": "Microsoft Graph",
                    "path": None,
                }
                _ = upsert_model_version(ver_meta, model_artifact_id=model_art.artifact_id)
                count_versions += 1

        return f"Copilot sync done. Models: {count_models}, Versions: {count_versions}."
