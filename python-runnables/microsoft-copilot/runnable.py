# python-runnables/microsoft-copilot/runnable.py
from dataiku.runnables import Runnable


class MyRunnable(Runnable):
    """
    Microsoft Copilot → Dataiku Govern sync (via Microsoft Graph)

    Creates/updates:
      - A Govern "Project" representing the Microsoft 365 tenant
      - One "External Model" shell per Copilot Product SKU (e.g., MICRSOFT_COPILOT, Copilot for M365)
      - One "External Model Version" per service plan under the SKU

    Notes:
      - This runnable uses client credentials against Microsoft Graph to list subscribed SKUs and service plans.
      - It doesn't pull private chat content. Only license metadata and plan details.
    """

    DEFAULTS = {
        "graph_endpoint": "https://graph.microsoft.com",
        "graph_use_beta": False,
        "COPILOT_ONLY_SERVICEPLAN_NAME": None,
    }

    def __init__(self, project_key, config, plugin_config):
        self.project_key = project_key
        self.config = config or {}
        self.plugin_config = plugin_config or {}

    def get_progress_target(self):
        return None

    def run(self, progress_callback):
        import os, json
        from urllib.parse import urlencode
        from datetime import datetime, timezone

        import requests
        import dataikuapi
        from dataikuapi.govern.blueprint import GovernBlueprintVersionId
        from dataikuapi.govern.artifact_search import (
            GovernArtifactSearchQuery,
            GovernArtifactFilterBlueprints,
            GovernArtifactFilterFieldValue,
        )

        # ----- config helpers -----
        def pick(k, *, alt=None):
            return (
                self.config.get(k)
                or self.plugin_config.get(k)
                or os.environ.get(alt or k)
                or self.DEFAULTS.get(alt or k)
            )

        TENANT_ID = pick("tenant_id")
        CLIENT_ID = pick("client_id")
        CLIENT_SECRET = pick("client_secret")
        GOVERN_BASE = pick("govern_base")
        GOVERN_APIKEY = pick("govern_API_key")

        GRAPH_BASE = pick("graph_endpoint") or "https://graph.microsoft.com"
        GRAPH_USE_BETA = bool(pick("graph_use_beta"))
        ONLY_PLAN_NAME = pick("COPILOT_ONLY_SERVICEPLAN_NAME")

        if not all([TENANT_ID, CLIENT_ID, CLIENT_SECRET, GOVERN_BASE, GOVERN_APIKEY]):
            raise ValueError("Missing required configuration: tenant_id, client_id, client_secret, govern_base, govern_API_key")

        # ----- Govern blueprint configuration (reuse generic external blueprints) -----
        BP_PARENT_ID, BV_PARENT_ID = "bp.system.govern_project", "bv.iso_42001_v05"
        BP_MODEL_ID, BV_MODEL_ID = "bp.external_models", "bv.azureml_model"  # a generic external model shell
        BP_VER_ID, BV_VER_ID = "bp.external_model_version", "bv.azureml_model_version"  # generic version

        F_EXT = "external_id"
        F_DESC = "description"
        F_TAGS = "tags_json"
        F_URL = "url_azureml"  # reuse field to store Microsoft admin URL

        F_VER = "version"
        F_CRTON = "creation_date"
        F_CRTBY = "created_by"
        F_LASTMOD = "last_modified"
        F_MODBY = "modified_by"
        F_PATH = "path"

        F_PROJ_MODELS = "govern_models"
        F_MODEL_PARENT = "govern_project"
        F_MODEL_CHILD = "govern_model_versions"
        F_VER_PARENT = "model"

        # ----- tiny helpers -----
        def _set_reference(defn, field_id: str, artifact_id: str, list_field: bool = False):
            raw = defn.get_raw()
            fields = raw.setdefault("fields", {})
            cur = fields.get(field_id)
            if list_field:
                if cur is None:
                    fields[field_id] = [artifact_id]; return
                if isinstance(cur, list):
                    ids = {(x.get("artifactId") if isinstance(x, dict) else x) for x in cur}
                    if artifact_id not in ids:
                        cur.append({"artifactId": artifact_id} if (cur and isinstance(cur[0], dict)) else artifact_id)
                    fields[field_id] = cur; return
                if isinstance(cur, dict) and "artifactId" in cur:
                    fields[field_id] = [cur] if cur.get("artifactId") == artifact_id else [cur, {"artifactId": artifact_id}]
                    return
                fields[field_id] = [cur] if cur == artifact_id else [cur, artifact_id]
                return
            # single
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
            q.add_artifact_filter(GovernArtifactFilterFieldValue("EQUALS", condition=value, field_id=field_id))
            hits = gc.new_artifact_search_request(q).fetch_next_batch(page_size=1).get_response_hits()
            return hits[0].to_artifact() if hits else None

        # ----- Govern ops -----
        def upsert_project(tenant_display_name: str):
            name = f"Microsoft 365 Tenant - {tenant_display_name}"
            # find by name
            q = GovernArtifactSearchQuery()
            q.add_artifact_filter(GovernArtifactFilterBlueprints([BP_PARENT_ID]))
            q.add_artifact_filter(GovernArtifactFilterFieldValue("EQUALS", condition=name))
            hits = gc.new_artifact_search_request(q).fetch_next_batch(page_size=1).get_response_hits()
            if hits:
                return hits[0].to_artifact(), "existing"
            payload = {
                "blueprintVersionId": GovernBlueprintVersionId(BP_PARENT_ID, BV_PARENT_ID).build(),
                "name": name,
                "fields": {},
            }
            return gc.create_artifact(payload), "created"

        def upsert_model_shell(meta: dict, project_artifact_id: str | None):
            art = find_one_by_field(BP_MODEL_ID, F_EXT, meta["external_id"])
            fields = {
                F_EXT: meta["external_id"],
                F_DESC: meta.get("description"),
                F_TAGS: json.dumps(meta.get("tags", {})),
                F_URL: meta.get("url_azureml"),
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            if art:
                d = art.get_definition(); raw = d.get_raw()
                raw.setdefault("fields", {}).update(fields)
                if raw.get("name") != meta["name"]:
                    raw["name"] = meta["name"]
                if project_artifact_id:
                    _set_reference(d, F_MODEL_PARENT, project_artifact_id, list_field=False)
                d.save(); return art, "updated"
            payload = {
                "blueprintVersionId": GovernBlueprintVersionId(BP_MODEL_ID, BV_MODEL_ID).build(),
                "name": meta["name"],
                "fields": fields,
            }
            art = gc.create_artifact(payload)
            if project_artifact_id:
                d = art.get_definition(); _set_reference(d, F_MODEL_PARENT, project_artifact_id, list_field=False); d.save()
            return art, "created"

        def link_model_into_project(project_art, model_artifact_id: str):
            d = project_art.get_definition(); raw = d.get_raw(); fields = raw.setdefault("fields", {})
            cur = fields.get(F_PROJ_MODELS)
            if cur is None:
                fields[F_PROJ_MODELS] = [model_artifact_id]; d.save(); return
            if isinstance(cur, list):
                ids = {(x.get("artifactId") if isinstance(x, dict) else x) for x in cur}
                if model_artifact_id not in ids:
                    cur.append({"artifactId": model_artifact_id} if (cur and isinstance(cur[0], dict)) else model_artifact_id)
                    fields[F_PROJ_MODELS] = cur; d.save(); return
            if isinstance(cur, dict) and "artifactId" in cur:
                if cur.get("artifactId") != model_artifact_id:
                    fields[F_PROJ_MODELS] = [cur, {"artifactId": model_artifact_id}]; d.save(); return
            if isinstance(cur, str) and cur != model_artifact_id:
                fields[F_PROJ_MODELS] = [cur, model_artifact_id]; d.save()

        def upsert_model_version(meta: dict, model_artifact_id: str | None):
            art = find_one_by_field(BP_VER_ID, F_EXT, meta["external_id"])
            fields = {
                F_EXT: meta["external_id"],
                F_VER: str(meta.get("version")),
                F_DESC: meta.get("description"),
                F_TAGS: json.dumps(meta.get("tags", {})),
                F_CRTON: meta.get("created_on"),
                F_CRTBY: meta.get("created_by"),
                F_URL: meta.get("url_azureml"),
                F_LASTMOD: meta.get("last_modified"),
                F_MODBY: meta.get("modified_by"),
                F_PATH: meta.get("path"),
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            title = f"{meta['name']} - v{meta.get('version') or '?'}"
            if art:
                d = art.get_definition(); raw = d.get_raw(); raw.setdefault("fields", {}).update(fields)
                if raw.get("name") != title:
                    raw["name"] = title
                if model_artifact_id:
                    _set_reference(d, F_VER_PARENT, model_artifact_id, list_field=False)
                d.save(); return art, "updated"
            payload = {
                "blueprintVersionId": GovernBlueprintVersionId(BP_VER_ID, BV_VER_ID).build(),
                "name": title,
                "fields": fields,
            }
            art = gc.create_artifact(payload)
            if model_artifact_id:
                d = art.get_definition(); _set_reference(d, F_VER_PARENT, model_artifact_id, list_field=False); d.save()
            return art, "created"

        # ----- Graph auth (client credentials) -----
        token_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
        scope = f"{GRAPH_BASE}/.default"
        resp = requests.post(
            token_url,
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "scope": scope,
                "grant_type": "client_credentials",
            },
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"Graph token error: {resp.status_code} {resp.text}")
        access_token = resp.json().get("access_token")
        if not access_token:
            raise RuntimeError("No access_token from Microsoft identity platform.")

        # ----- API clients -----
        gc = dataikuapi.GovernClient(GOVERN_BASE.rstrip("/"), GOVERN_APIKEY, insecure_tls=True)

        # ----- Graph calls: organization + subscribedSkus -----
        api_ver = "beta" if GRAPH_USE_BETA else "v1.0"
        headers = {"Authorization": f"Bearer {access_token}"}

        org = requests.get(f"{GRAPH_BASE}/{api_ver}/organization", headers=headers, timeout=30)
        org.raise_for_status()
        org_value = (org.json() or {}).get("value", [])
        tenant_display_name = (org_value[0] or {}).get("displayName") if org_value else TENANT_ID

        # project representing tenant
        project_art, _ = upsert_project(tenant_display_name)

        # list SKUs
        skus = requests.get(f"{GRAPH_BASE}/{api_ver}/subscribedSkus", headers=headers, timeout=60)
        skus.raise_for_status()
        sku_items = (skus.json() or {}).get("value", [])

        count_models, count_versions = 0, 0
        for sku in sku_items:
            sku_id = sku.get("skuId")
            sku_part = sku.get("skuPartNumber")  # e.g., MICROSOFT_COPILOT or M365_COPILOT
            service_plans = sku.get("servicePlans", []) or []

            # We treat each Copilot SKU as a model shell
            if not sku_part or "COPILOT" not in sku_part.upper():
                continue

            admin_url = "https://admin.microsoft.com/#/subscriptions"  # generic admin portal link
            model_meta = {
                "external_id": f"m365:sku:{sku_id}",
                "name": f"Microsoft Copilot - {sku_part}",
                "description": "Microsoft Copilot subscription SKU discovered via Microsoft Graph subscribedSkus",
                "tags": {
                    "cap_total_units": sku.get("capabilityStatus"),
                    "prepaid_units": json.dumps(sku.get("prepaidUnits", {})),
                    "consumed_units": sku.get("consumedUnits"),
                },
                "url_azureml": admin_url,
            }
            model_art, _ = upsert_model_shell(model_meta, project_artifact_id=project_art.artifact_id)
            link_model_into_project(project_art, model_art.artifact_id)
            count_models += 1

            # create a version per service plan under the SKU
            for sp in service_plans:
                sp_name = sp.get("servicePlanName")
                if ONLY_PLAN_NAME and sp_name != ONLY_PLAN_NAME:
                    continue
                ver = sp.get("servicePlanId")  # GUID
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
                    "url_azureml": admin_url,
                    "created_on": None,
                    "created_by": None,
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                    "modified_by": "Microsoft Graph",
                    "path": None,
                }
                ver_art, _ = upsert_model_version(ver_meta, model_artifact_id=model_art.artifact_id)
                count_versions += 1

        return f"Copilot sync done. Models: {count_models}, Versions: {count_versions}."
