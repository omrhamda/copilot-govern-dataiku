from dataiku.runnables import Runnable


class MyRunnable(Runnable):
    def __init__(self, project_key, config, plugin_config):
        self.project_key = project_key
        self.config = config or {}
        self.plugin_config = plugin_config or {}

    def get_progress_target(self):
        return None

    def run(self, progress_callback):
        return "This runnable is deprecated and has been removed."

        def find_parent_by_name(name: str):
            q = GovernArtifactSearchQuery()
            q.add_artifact_filter(GovernArtifactFilterBlueprints([BP_PARENT_ID]))
            q.add_artifact_filter(GovernArtifactFilterFieldValue("EQUALS", condition=name))
            hits = gc.new_artifact_search_request(q).fetch_next_batch(page_size=1).get_response_hits()
            return hits[0].to_artifact() if hits else None

        def upsert_parent_project(workspace_name: str):
            name = f"AzureML Project - {workspace_name}"
            art = find_parent_by_name(name)
            if art:
                return art, "existing"
            payload = {
                "blueprintVersionId": GovernBlueprintVersionId(BP_PARENT_ID, BV_PARENT_ID).build(),
                "name": name,
                "fields": {},
            }
            return gc.create_artifact(payload), "created"

        # ---------- Model shell ----------
        def model_external_id_from_version_id(version_id: str) -> str:
            """Turn azureml .../models/{name}/versions/{ver} → .../models/{name}"""
            return re.sub(r"/versions/[^/]+$", "", version_id)

        def upsert_model_shell(meta: dict, project_artifact_id: str | None):
            art = find_one_by_field(BP_M_ID, F_EXT_M, meta["external_id"])
            fields = {
                F_EXT_M:  meta["external_id"],
                F_DESC_M: meta.get("description"),
                F_TAGS_M: json.dumps(meta.get("tags", {})),
                F_URL_M:  meta.get("url_azureml"),
            }
            fields = {k: v for k, v in fields.items() if v is not None}

            if art:
                d = art.get_definition()
                raw = d.get_raw()
                raw.setdefault("fields", {}).update(fields)
                if raw.get("name") != meta["name"]:
                    raw["name"] = meta["name"]  # name w/o version
                if project_artifact_id:
                    _set_reference(d, F_MODEL_PARENT, project_artifact_id, list_field=False)
                d.save()
                return art, "updated"

            payload = {
                "blueprintVersionId": GovernBlueprintVersionId(BP_M_ID, BV_M_ID).build(),
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
                fields[F_PROJ_MODELS] = [model_artifact_id]; d.save(); return

            if isinstance(cur, list):
                ids = {(x.get("artifactId") if isinstance(x, dict) else x) for x in cur}
                if model_artifact_id not in ids:
                    use_objects = bool(cur and isinstance(cur[0], dict))
                    cur.append({"artifactId": model_artifact_id} if use_objects else model_artifact_id)
                    fields[F_PROJ_MODELS] = cur; d.save(); return

            if isinstance(cur, dict) and "artifactId" in cur:
                if cur.get("artifactId") != model_artifact_id:
                    fields[F_PROJ_MODELS] = [cur, {"artifactId": model_artifact_id}]
                    d.save(); return

            if isinstance(cur, str) and cur != model_artifact_id:
                fields[F_PROJ_MODELS] = [cur, model_artifact_id]; d.save()

        def link_version_under_model(model_art, version_artifact_id: str):
            d = model_art.get_definition()
            try:
                _set_reference(d, F_MODEL_CHILD, version_artifact_id, list_field=True)
                d.save()
            except DataikuException as e:
                # tolerate schema where field is single-valued
                if "is not a list" not in str(e):
                    raise
                d = model_art.get_definition()
                _set_reference(d, F_MODEL_CHILD, version_artifact_id, list_field=False)
                d.save()

        # ---------- Model version ----------
        def upsert_model_version(meta: dict, model_artifact_id: str | None):
            art = find_one_by_field(BP_VER_ID, F_EXT_MV, meta["external_id"])
            fields = {
                F_EXT_MV: meta["external_id"],
                F_VER_MV: str(meta.get("version")),
                F_DESC_MV: meta.get("description"),
                F_TAGS_MV: json.dumps(meta.get("tags", {})),
                F_CRTON_MV: meta.get("created_on"),
                F_CRTBY_MV: meta.get("created_by"),
                F_URL_MV: meta.get("url_azureml"),
                F_LASTMOD_MV: meta.get("last_modified"),
                F_MODBY_MV:   meta.get("modified_by"),
                F_PATH_MV:    meta.get("path"),
                F_MODEL_CARD_MV: meta.get("model_card"),
                F_APPROVAL_MV:   meta.get("approval_status"),
                F_RISK_MV:       meta.get("risk_assessment"),
                F_BIAS_MV:       meta.get("bias_metrics"),
                F_PERF_MV:       meta.get("performance_metrics"),
                F_EXPL_MV:       meta.get("explainability"),
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            title = _title_with_version(meta["name"], meta.get("version"))

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

        # ---------- Dataset ----------
        def upsert_dataset(ds_meta: dict, model_version_artifact_id: str | None):
            art = find_one_by_field(BP_DS_ID, F_EXT_D, ds_meta["external_id"])

            fields = {
                F_EXT_D: ds_meta["external_id"],
                F_VER_D: str(ds_meta.get("version")),
                F_DESC_D: ds_meta.get("description"),
                F_TAGS_D: json.dumps(ds_meta.get("tags", {})),
                F_CRTON_D: ds_meta.get("created_on"),
                F_CRTBY_D: ds_meta.get("created_by"),
                F_URL_D: ds_meta.get("url_azureml"),
                F_LASTMOD_D: ds_meta.get("last_modified"),
                F_MODBY_D:   ds_meta.get("modified_by"),
                F_PATH_D:    ds_meta.get("path"),
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            title = _title_with_version(ds_meta["name"], ds_meta.get("version"))

            if art:
                d = art.get_definition()
                raw = d.get_raw()
                raw.setdefault("fields", {}).update(fields)
                if raw.get("name") != title:
                    raw["name"] = title
                if model_version_artifact_id:
                    _set_reference(d, F_DS_LINK_MODEL, model_version_artifact_id, list_field=False)
                d.save()
                return art, "updated"

            payload = {
                "blueprintVersionId": GovernBlueprintVersionId(BP_DS_ID, BV_DS_ID).build(),
                "name": title,
                "fields": fields,
            }
            art = gc.create_artifact(payload)
            if model_version_artifact_id:
                d = art.get_definition()
                _set_reference(d, F_DS_LINK_MODEL, model_version_artifact_id, list_field=False)
                d.save()
            return art, "created"

        # ---------- AML helpers ----------
        def infer_job_name_from_model(model):
            p = getattr(model, "path", "") or ""
            m = re.search(r"/dcid\.([^/]+)/model$", p)
            return m.group(1) if m else getattr(model, "job_name", None)

        def resolve_input_to_data_asset(ml: MLClient, ref_or_obj):
            for attr in ("path", "uri", "uri_file", "uri_folder"):
                v = getattr(ref_or_obj, attr, None)
                if v: break
            else:
                v = str(ref_or_obj)
            m = re.fullmatch(r"(?:azureml:)?([^:]+):(.+)", v.strip())
            if not m:
                return v, None
            name, version = m.group(1), m.group(2)
            try:
                return v, ml.data.get(name=name, version=version)
            except Exception:
                return v, None

        def iter_models_in_workspace(ml_client: MLClient, only_name=None, only_ver=None):
            if only_name and only_ver:
                yield ml_client.models.get(name=only_name, version=only_ver); return
            if only_name:
                for mv in ml_client.models.list(name=only_name):
                    yield ml_client.models.get(name=only_name, version=mv.version); return
            seen = set()
            for m in ml_client.models.list():
                if m.name in seen: continue
                seen.add(m.name)
                for mv in ml_client.models.list(name=m.name):
                    yield ml_client.models.get(name=m.name, version=mv.version)

        # ---------- authenticate clients ----------
        cred = ClientSecretCredential(TENANT_ID, CLIENT_ID, CLIENT_SECRET)
        gc   = dataikuapi.GovernClient(GOVERN_BASE.rstrip("/"), GOVERN_APIKEY, insecure_tls=True)

        # ---------- discover workspaces & sync ----------
        summary = []

        # subscriptions
        if ONLY_SUB_ID:
            subs = [ONLY_SUB_ID]
        else:
            subs = [s.subscription_id for s in SubClient(cred).subscriptions.list()]

        print("Using subscription client:", SUB_CLIENT_SRC)
        print("Subscriptions:", subs)

        for sub_id in subs:
            ws_client = AzureMachineLearningWorkspaces(cred, sub_id)
            workspaces = list(ws_client.workspaces.list_by_subscription())
            print(sub_id, "→", [w.name for w in workspaces])

            for ws in workspaces:
                rg  = ws.id.split("/resourceGroups/")[1].split("/")[0]
                wsn = ws.name

                print(f"\n== Workspace: {wsn} (rg={rg}, sub={sub_id}) ==")
                ml = MLClient(cred, sub_id, rg, wsn)

                # Project
                project_art, project_status = upsert_parent_project(wsn)
                print(f"PROJECT {project_status.upper()}: {project_art}")

                # Models & versions
                for m in iter_models_in_workspace(ml, ONLY_NAME, ONLY_VER):
                    # model shell
                    model_url = (
                        f"https://ml.azure.com/model/{quote(m.name)}/versions"
                        f"?wsid=/subscriptions/{sub_id}/resourceGroups/{rg}/workspaces/{wsn}{TENANT_QP}"
                    )
                    model_meta = {
                        "external_id": model_external_id_from_version_id(m.id),
                        "name":        m.name,
                        "description": m.description,
                        "tags":        m.tags or {},
                        "url_azureml": model_url,
                    }
                    model_art, model_action = upsert_model_shell(model_meta, project_artifact_id=project_art.artifact_id)
                    link_model_into_project(project_art, model_art.artifact_id)
                    print(f"MODEL   {model_action.upper():7} {m.name}")

                    # model version
                    created_on = created_by = None
                    ctx = getattr(m, "creation_context", None)
                    if ctx and getattr(ctx, "created_at", None):
                        try:
                            created_on = ctx.created_at.astimezone(timezone.utc).date().isoformat()
                        except Exception:
                            created_on = None
                        created_by = getattr(ctx, "created_by", None)

                    nv = quote(f"{m.name}:{m.version}", safe=":")
                    aml_ver_url = (
                        f"https://ml.azure.com/model/{nv}/details"
                        f"?wsid=/subscriptions/{sub_id}/resourceGroups/{rg}/workspaces/{wsn}{TENANT_QP}#overview"
                    )
                    last_mod = getattr(ctx, "last_modified_at", None) if ctx else None

                    ver_meta = {
                        "external_id": m.id,
                        "name":        m.name,
                        "version":     m.version,
                        "description": m.description,
                        "tags":        m.tags or {},
                        "created_on":  created_on,
                        "created_by":  created_by,
                        "url_azureml": aml_ver_url,
                        "last_modified": last_mod.astimezone(timezone.utc).isoformat() if last_mod else None,
                        "modified_by":   getattr(ctx, "last_modified_by", None) if ctx else None,
                        "path":         getattr(m, "path", None),
                    }
                    for k in ["model_card","approval_status","risk_assessment","bias_metrics","performance_metrics","explainability"]:
                        v = (m.tags or {}).get(k)
                        if v is not None:
                            ver_meta[k] = v

                    ver_art, ver_action = upsert_model_version(ver_meta, model_artifact_id=model_art.artifact_id)
                    link_version_under_model(model_art, ver_art.artifact_id)
                    print(f"VERSION {ver_action.upper():7} {m.name} v{m.version}")
                    summary.append((wsn, m.name, str(m.version), ver_action))

                    # datasets (attached to the version)
                    job_name = infer_job_name_from_model(m)
                    if not job_name:
                        continue

                    job = ml.jobs.get(job_name)
                    inputs = getattr(job, "inputs", {}) or {}
                    wsid = f"/subscriptions/{sub_id}/resourceGroups/{rg}/workspaces/{wsn}"

                    d_ver = ver_art.get_definition()  # save once at end

                    for key, inp in inputs.items():
                        if not hasattr(inp, 'type') or inp.type not in [AssetTypes.URI_FILE, AssetTypes.URI_FOLDER, AssetTypes.MLTABLE]:
                                print(f"  SKIPPING input '{key}': Not a data asset (type: {getattr(inp, 'type', 'N/A')}).")
                                continue
                        ref, da = resolve_input_to_data_asset(ml, inp)
                        if not da:
                            continue

                        try:
                            created_on = getattr(da, "creation_context", None)
                            created_on = created_on.created_at.astimezone(timezone.utc).date().isoformat() if (created_on and created_on.created_at) else None
                            created_by = getattr(da.creation_context, "created_by", None) if getattr(da, "creation_context", None) else None
                            last_mod   = getattr(da, "creation_context", None)
                            last_mod   = last_mod.last_modified_at.astimezone(timezone.utc).isoformat() if (last_mod and last_mod.last_modified_at) else None
                            modified_by = getattr(da.creation_context, "last_modified_by", None) if getattr(da, "creation_context", None) else None
                        except Exception:
                            created_on = created_by = last_mod = modified_by = None

                        data_url = (
                            f"https://ml.azure.com/data/{quote(da.name)}/versions/{da.version}/details"
                            f"?wsid={wsid}{TENANT_QP}"
                        )

                        ds_meta = {
                            "external_id": da.id,
                            "name":        da.name,
                            "version":     da.version,
                            "description": getattr(da, "description", None),
                            "tags":        getattr(da, "tags", None) or {},
                            "created_on":  created_on,
                            "created_by":  created_by,
                            "last_modified": last_mod,
                            "modified_by":   modified_by,
                            "path":        getattr(da, "path", None),
                            "url_azureml": data_url,
                        }

                        ds_art, ds_action = upsert_dataset(ds_meta, model_version_artifact_id=ver_art.artifact_id)
                        _set_reference(d_ver, F_MODEL_DATASETS, ds_art.artifact_id, list_field=True)
                        print(f"  {ds_action.upper():7} DATASET {da.name}:{da.version} → linked to version {m.name}:{m.version}")

                    d_ver.save()

        # return a compact summary as the runnable’s output
        return "Sync completed. Check the job log for per-item details."

# # python-runnables/azure-ml/runnable.py
# from dataiku.runnables import Runnable

# class MyRunnable(Runnable):
#     """
#     Azure ML → Dataiku Govern sync
#     Project → Model (shell) → Model Version(s) + Datasets
#     """

#     # ---------- minimal, safe config resolver ----------
#     DEFAULTS = {
#         # set hardcoded values here if you prefer (leave None to require env/params)
#         "TENANT_ID": "",
#         "CLIENT_ID": "",
#         "CLIENT_SECRET": "",
#         "GOVERN_BASE": "",
#         "GOVERN_API_KEY": "",
#         # Optional filters
#         "AML_MODEL_NAME": None,
#         "AML_MODEL_VERSION": None,
#         "SUBSCRIPTION_ID": "",
#     }

#     def __init__(self, project_key, config, plugin_config):
#         self.project_key = project_key
#         self.config = config or {}
#         self.plugin_config = plugin_config or {}

#     def get_progress_target(self):
#         return None

#     # ------------------------- main run -------------------------
#     def run(self, progress_callback):
#         # --- tame noisy OpenTelemetry logs ---
#         import logging, warnings
#         for n in ("opentelemetry", "opentelemetry.instrumentation", "azure.ai.ml"):
#             logging.getLogger(n).setLevel(logging.ERROR)
#         warnings.filterwarnings("ignore", message="Attempting to instrument")

#         # ---------- imports ----------
#         import os, re, json
#         from datetime import datetime, timezone
#         from urllib.parse import quote

#         from azure.identity import ClientSecretCredential

#         # Prefer new subscription SDK, fallback to legacy path
#         try:
#             from azure.mgmt.subscription import SubscriptionClient as SubClient
#             SUB_CLIENT_SRC = "azure-mgmt-subscription"
#         except Exception:
#             from azure.mgmt.resource.subscriptions import SubscriptionClient as SubClient
#             SUB_CLIENT_SRC = "azure-mgmt-resource.subscriptions"

#         from azure.mgmt.machinelearningservices import AzureMachineLearningWorkspaces
#         from azure.ai.ml import MLClient
#         from azure.ai.ml.constants._common import AssetTypes

#         import dataikuapi
#         from dataikuapi.govern.blueprint import GovernBlueprintVersionId
#         from dataikuapi.govern.artifact_search import (
#             GovernArtifactSearchQuery,
#             GovernArtifactFilterBlueprints,
#             GovernArtifactFilterFieldValue,
#         )
#         from dataikuapi.utils import DataikuException

#         # ---------- resolve configuration ----------
#         def pick(key_env_or_param, *, alt_env=None):
#             # order: runnable param → plugin param → env var → DEFAULTS
#             return (
#                 self.config.get(key_env_or_param)
#                 or self.plugin_config.get(key_env_or_param)
#                 or os.environ.get(alt_env or key_env_or_param)
#                 or self.DEFAULTS.get(alt_env or key_env_or_param)
#             )
        
#         TENANT_ID     = pick("tenant_id")
#         CLIENT_ID     = pick("client_id")
#         CLIENT_SECRET = pick("client_secret")
#         GOVERN_BASE   = pick("govern_base")
#         GOVERN_APIKEY = pick("govern_API_key")

# #         TENANT_ID     = pick("tenant_id",     alt_env="TENANT_ID")
# #         CLIENT_ID     = pick("client_id",     alt_env="CLIENT_ID")
# #         CLIENT_SECRET = pick("client_secret", alt_env="CLIENT_SECRET")
# #         GOVERN_BASE   = pick("govern_base",   alt_env="GOVERN_BASE")
# #         GOVERN_APIKEY = pick("govern_API_key",alt_env="GOVERN_API_KEY")

#         # optional filters
#         ONLY_NAME     = pick("AML_MODEL_NAME")
#         ONLY_VER      = pick("AML_MODEL_VERSION")
#         ONLY_SUB_ID   = pick("SUBSCRIPTION_ID")  # restrict to one subscription when set

#         # fail early if required secrets are missing
#         missing = [k for k, v in {
#             "TENANT_ID": TENANT_ID, "CLIENT_ID": CLIENT_ID, "CLIENT_SECRET": CLIENT_SECRET,
#             "GOVERN_BASE": GOVERN_BASE, "GOVERN_API_KEY": GOVERN_APIKEY
#         }.items() if not v]
#         if missing:
#             raise ValueError("Missing required configuration: " + ", ".join(missing))

#         TENANT_QP = f"&tid={TENANT_ID}" if TENANT_ID else ""

#         # ---------- Govern blueprint IDs ----------
#         # Project
#         BP_PARENT_ID, BV_PARENT_ID = "bp.system.govern_project", "bv.azureml_project"

#         # Model shell (middle layer)
#         BP_M_ID, BV_M_ID = "bp.external_model", "bv.azureml_model"

#         # Model version
#         BP_VER_ID, BV_VER_ID = "bp.external_model_version", "bv.azureml_model_version"

#         # Dataset
#         BP_DS_ID, BV_DS_ID = "bp.governed_dataset", "bv.azureml_dataset"

#          # ---------- model version govern fields ----------
#         F_EXT_MV, F_VER_MV, F_DESC_MV, F_TAGS_MV = "external_id", "version", "description", "tags_json"
#         F_CRTON_MV, F_CRTBY_MV, F_URL_MV      = "creation_date", "created_by", "url_azureml"
#         F_LASTMOD_MV, F_MODBY_MV, F_PATH_MV   = "last_modified", "modified_by", "path"
        
#         # optional extra govern fields on model version
#         F_MODEL_CARD_MV   = "tag_model_card"
#         F_APPROVAL_MV     = "tag_approval_status"
#         F_RISK_MV         = "tag_risk_assessment"
#         F_BIAS_MV         = "tag_bias_metrics"
#         F_PERF_MV         = "tag_performance_metrics"
#         F_EXPL_MV         = "tag_explainability"      
                
#         # ---------- model govern fields ----------
#         F_EXT_M, F_DESC_M, F_TAGS_M = "external_id", "description", "tags_json"
#         F_URL_M = "url_azureml"
        
#         # ---------- dataset govern fields ----------
#         F_EXT, F_VER, F_DESC, F_TAGS = "external_id", "version", "description", "tags_json"
#         F_CRTON, F_CRTBY, F_URL      = "created_on", "created_by", "url_azureml"
#         F_LASTMOD, F_MODBY, F_PATH   = "last_modified", "modified_by", "path"


#         # relationships
#         F_PROJ_MODELS     = "models"               # project → [models]
#         F_MODEL_PARENT    = "govern_project"       # model   → project
#         F_MODEL_CHILD     = "govern_model_versions"  # model   → [versions]
#         F_VER_PARENT      = "model"          # version → model
#         F_MODEL_DATASETS  = "governed_dataset"     # version → [datasets]
#         F_DS_LINK_MODEL   = "link_model"           # dataset → version (single in your BV)

#         # ---------- helpers ----------
#         def _title_with_version(name, version):
#             v = str(version).lstrip("vV") if version is not None else "?"
#             return f"{name} - v{v}"

#         def _set_reference(defn, field_id: str, artifact_id: str, list_field: bool = False):
#             """Safely set/append a reference field on a Govern artifact definition."""
#             raw = defn.get_raw()
#             fields = raw.setdefault("fields", {})
#             cur = fields.get(field_id)

#             if list_field:
#                 if cur is None:
#                     fields[field_id] = [artifact_id]; return
#                 if isinstance(cur, list):
#                     use_objects = bool(cur and isinstance(cur[0], dict))
#                     existing_ids = {(x.get("artifactId") if isinstance(x, dict) else x) for x in cur}
#                     if artifact_id not in existing_ids:
#                         cur.append({"artifactId": artifact_id} if use_objects else artifact_id)
#                     fields[field_id] = cur; return
#                 if isinstance(cur, dict) and "artifactId" in cur:
#                     existing_id = cur.get("artifactId")
#                     fields[field_id] = [cur] if existing_id == artifact_id else [cur, {"artifactId": artifact_id}]
#                     return
#                 fields[field_id] = [cur] if cur == artifact_id else [cur, artifact_id]
#                 return

#             # single-valued reference
#             if isinstance(cur, list):
#                 if artifact_id not in cur:
#                     cur.append(artifact_id)
#                 fields[field_id] = cur
#             elif isinstance(cur, dict):
#                 fields[field_id] = {"artifactId": artifact_id}
#             else:
#                 fields[field_id] = artifact_id

#         def find_one_by_field(bp_id: str, field_id: str, value: str):
#             q = GovernArtifactSearchQuery()
#             q.add_artifact_filter(GovernArtifactFilterBlueprints([bp_id]))
#             q.add_artifact_filter(GovernArtifactFilterFieldValue("EQUALS", condition=value, field_id=field_id))
#             hits = gc.new_artifact_search_request(q).fetch_next_batch(page_size=1).get_response_hits()
#             return hits[0].to_artifact() if hits else None

#         def find_parent_by_name(name: str):
#             q = GovernArtifactSearchQuery()
#             q.add_artifact_filter(GovernArtifactFilterBlueprints([BP_PARENT_ID]))
#             q.add_artifact_filter(GovernArtifactFilterFieldValue("EQUALS", condition=name))
#             hits = gc.new_artifact_search_request(q).fetch_next_batch(page_size=1).get_response_hits()
#             return hits[0].to_artifact() if hits else None

#         def upsert_parent_project(workspace_name: str):
#             name = f"AzureML Project - {workspace_name}"
#             art = find_parent_by_name(name)
#             if art:
#                 return art, "existing"
#             payload = {
#                 "blueprintVersionId": GovernBlueprintVersionId(BP_PARENT_ID, BV_PARENT_ID).build(),
#                 "name": name,
#                 "fields": {},
#             }
#             return gc.create_artifact(payload), "created"

#         # ---------- Model shell ----------
#         def model_external_id_from_version_id(version_id: str) -> str:
#             """Turn azureml .../models/{name}/versions/{ver} → .../models/{name}"""
#             return re.sub(r"/versions/[^/]+$", "", version_id)

#         def upsert_model_shell(meta: dict, project_artifact_id: str | None):
#             art = find_one_by_field(BP_M_ID, F_EXT, meta["external_id"])
#             fields = {
#                 F_EXT:  meta["external_id"],
#                 F_DESC: meta.get("description"),
#                 F_TAGS: json.dumps(meta.get("tags", {})),
#                 F_URL:  meta.get("url_azureml"),
#             }
#             fields = {k: v for k, v in fields.items() if v is not None}

#             if art:
#                 d = art.get_definition()
#                 raw = d.get_raw()
#                 raw.setdefault("fields", {}).update(fields)
#                 if raw.get("name") != meta["name"]:
#                     raw["name"] = meta["name"]  # name w/o version
#                 if project_artifact_id:
#                     _set_reference(d, F_MODEL_PARENT, project_artifact_id, list_field=False)
#                 d.save()
#                 return art, "updated"

#             payload = {
#                 "blueprintVersionId": GovernBlueprintVersionId(BP_M_ID, BV_M_ID).build(),
#                 "name": meta["name"],
#                 "fields": fields,
#             }
#             art = gc.create_artifact(payload)
#             if project_artifact_id:
#                 d = art.get_definition()
#                 _set_reference(d, F_MODEL_PARENT, project_artifact_id, list_field=False)
#                 d.save()
#             return art, "created"

#         def link_model_into_project(project_art, model_artifact_id: str):
#             d = project_art.get_definition()
#             raw = d.get_raw()
#             fields = raw.setdefault("fields", {})
#             cur = fields.get(F_PROJ_MODELS)

#             if cur is None:
#                 fields[F_PROJ_MODELS] = [model_artifact_id]; d.save(); return

#             if isinstance(cur, list):
#                 ids = {(x.get("artifactId") if isinstance(x, dict) else x) for x in cur}
#                 if model_artifact_id not in ids:
#                     use_objects = bool(cur and isinstance(cur[0], dict))
#                     cur.append({"artifactId": model_artifact_id} if use_objects else model_artifact_id)
#                     fields[F_PROJ_MODELS] = cur; d.save(); return

#             if isinstance(cur, dict) and "artifactId" in cur:
#                 if cur.get("artifactId") != model_artifact_id:
#                     fields[F_PROJ_MODELS] = [cur, {"artifactId": model_artifact_id}]
#                     d.save(); return

#             if isinstance(cur, str) and cur != model_artifact_id:
#                 fields[F_PROJ_MODELS] = [cur, model_artifact_id]; d.save()

#         def link_version_under_model(model_art, version_artifact_id: str):
#             d = model_art.get_definition()
#             try:
#                 _set_reference(d, F_MODEL_CHILD, version_artifact_id, list_field=True)
#                 d.save()
#             except DataikuException as e:
#                 # tolerate schema where field is single-valued
#                 if "is not a list" not in str(e):
#                     raise
#                 d = model_art.get_definition()
#                 _set_reference(d, F_MODEL_CHILD, version_artifact_id, list_field=False)
#                 d.save()

#         # ---------- Model version ----------
#         def upsert_model_version(meta: dict, model_artifact_id: str | None):
#             art = find_one_by_field(BP_VER_ID, F_EXT, meta["external_id"])
#             fields = {
#                 F_EXT_MV: meta["external_id"],
#                 F_VER_MV: str(meta.get("version")),
#                 F_DESC_MV: meta.get("description"),
#                 F_TAGS_MV: json.dumps(meta.get("tags", {})),
#                 F_CRTON_MV: meta.get("created_on"),
#                 F_CRTBY_MV: meta.get("created_by"),
#                 F_URL_MV: meta.get("url_azureml"),
#                 F_LASTMOD_MV: meta.get("last_modified"),
#                 F_MODBY_MV:   meta.get("modified_by"),
#                 F_PATH_MV:    meta.get("path"),
#                 F_MODEL_CARD_MV: meta.get("model_card"),
#                 F_APPROVAL_MV:   meta.get("approval_status"),
#                 F_RISK_MV:       meta.get("risk_assessment"),
#                 F_BIAS_MV:       meta.get("bias_metrics"),
#                 F_PERF_MV:       meta.get("performance_metrics"),
#                 F_EXPL_MV:       meta.get("explainability"),
#             }
#             fields = {k: v for k, v in fields.items() if v is not None}
#             title = _title_with_version(meta["name"], meta.get("version"))

#             if art:
#                 d = art.get_definition()
#                 raw = d.get_raw()
#                 raw.setdefault("fields", {}).update(fields)
#                 if raw.get("name") != title:
#                     raw["name"] = title
#                 if model_artifact_id:
#                     _set_reference(d, F_VER_PARENT, model_artifact_id, list_field=False)
#                 d.save()
#                 return art, "updated"

#             payload = {
#                 "blueprintVersionId": GovernBlueprintVersionId(BP_VER_ID, BV_VER_ID).build(),
#                 "name": title,
#                 "fields": fields,
#             }
#             art = gc.create_artifact(payload)
#             if model_artifact_id:
#                 d = art.get_definition()
#                 _set_reference(d, F_VER_PARENT, model_artifact_id, list_field=False)
#                 d.save()
#             return art, "created"

#         # ---------- Dataset ----------
#         def upsert_dataset(ds_meta: dict, model_version_artifact_id: str | None):
#             art = find_one_by_field(BP_DS_ID, F_EXT, ds_meta["external_id"])

#             fields = {
#                 F_EXT: ds_meta["external_id"],
#                 F_VER: str(ds_meta.get("version")),
#                 F_DESC: ds_meta.get("description"),
#                 F_TAGS: json.dumps(ds_meta.get("tags", {})),
#                 F_CRTON: ds_meta.get("created_on"),
#                 F_CRTBY: ds_meta.get("created_by"),
#                 F_URL: ds_meta.get("url_azureml"),
#                 F_LASTMOD: ds_meta.get("last_modified"),
#                 F_MODBY:   ds_meta.get("modified_by"),
#                 F_PATH:    ds_meta.get("path"),
#             }
#             fields = {k: v for k, v in fields.items() if v is not None}
#             title = _title_with_version(ds_meta["name"], ds_meta.get("version"))

#             if art:
#                 d = art.get_definition()
#                 raw = d.get_raw()
#                 raw.setdefault("fields", {}).update(fields)
#                 if raw.get("name") != title:
#                     raw["name"] = title
#                 if model_version_artifact_id:
#                     _set_reference(d, F_DS_LINK_MODEL, model_version_artifact_id, list_field=False)
#                 d.save()
#                 return art, "updated"

#             payload = {
#                 "blueprintVersionId": GovernBlueprintVersionId(BP_DS_ID, BV_DS_ID).build(),
#                 "name": title,
#                 "fields": fields,
#             }
#             art = gc.create_artifact(payload)
#             if model_version_artifact_id:
#                 d = art.get_definition()
#                 _set_reference(d, F_DS_LINK_MODEL, model_version_artifact_id, list_field=False)
#                 d.save()
#             return art, "created"

#         # ---------- AML helpers ----------
#         def infer_job_name_from_model(model):
#             p = getattr(model, "path", "") or ""
#             m = re.search(r"/dcid\.([^/]+)/model$", p)
#             return m.group(1) if m else getattr(model, "job_name", None)

#         def resolve_input_to_data_asset(ml: MLClient, ref_or_obj):
#             for attr in ("path", "uri", "uri_file", "uri_folder"):
#                 v = getattr(ref_or_obj, attr, None)
#                 if v: break
#             else:
#                 v = str(ref_or_obj)
#             m = re.fullmatch(r"(?:azureml:)?([^:]+):(.+)", v.strip())
#             if not m:
#                 return v, None
#             name, version = m.group(1), m.group(2)
#             try:
#                 return v, ml.data.get(name=name, version=version)
#             except Exception:
#                 return v, None

#         def iter_models_in_workspace(ml_client: MLClient, only_name=None, only_ver=None):
#             if only_name and only_ver:
#                 yield ml_client.models.get(name=only_name, version=only_ver); return
#             if only_name:
#                 for mv in ml_client.models.list(name=only_name):
#                     yield ml_client.models.get(name=only_name, version=mv.version); return
#             seen = set()
#             for m in ml_client.models.list():
#                 if m.name in seen: continue
#                 seen.add(m.name)
#                 for mv in ml_client.models.list(name=m.name):
#                     yield ml_client.models.get(name=m.name, version=mv.version)

#         # ---------- authenticate clients ----------
#         cred = ClientSecretCredential(TENANT_ID, CLIENT_ID, CLIENT_SECRET)
#         gc   = dataikuapi.GovernClient(GOVERN_BASE.rstrip("/"), GOVERN_APIKEY, insecure_tls=True)

#         # ---------- discover workspaces & sync ----------
#         summary = []

#         # subscriptions
#         if ONLY_SUB_ID:
#             subs = [ONLY_SUB_ID]
#         else:
#             subs = [s.subscription_id for s in SubClient(cred).subscriptions.list()]

#         print("Using subscription client:", SUB_CLIENT_SRC)
#         print("Subscriptions:", subs)

#         for sub_id in subs:
#             ws_client = AzureMachineLearningWorkspaces(cred, sub_id)
#             workspaces = list(ws_client.workspaces.list_by_subscription())
#             print(sub_id, "→", [w.name for w in workspaces])

#             for ws in workspaces:
#                 rg  = ws.id.split("/resourceGroups/")[1].split("/")[0]
#                 wsn = ws.name

#                 print(f"\n== Workspace: {wsn} (rg={rg}, sub={sub_id}) ==")
#                 ml = MLClient(cred, sub_id, rg, wsn)

#                 # Project
#                 project_art, project_status = upsert_parent_project(wsn)
#                 print(f"PROJECT {project_status.upper()}: {project_art}")

#                 # Models & versions
#                 for m in iter_models_in_workspace(ml, ONLY_NAME, ONLY_VER):
#                     # model shell
#                     model_url = (
#                         f"https://ml.azure.com/model/{quote(m.name)}/versions"
#                         f"?wsid=/subscriptions/{sub_id}/resourceGroups/{rg}/workspaces/{wsn}{TENANT_QP}"
#                     )
#                     model_meta = {
#                         "external_id": model_external_id_from_version_id(m.id),
#                         "name":        m.name,
#                         "description": m.description,
#                         "tags":        m.tags or {},
#                         "url_azureml": model_url,
#                     }
#                     model_art, model_action = upsert_model_shell(model_meta, project_artifact_id=project_art.artifact_id)
#                     link_model_into_project(project_art, model_art.artifact_id)
#                     print(f"MODEL   {model_action.upper():7} {m.name}")

#                     # model version
#                     created_on = created_by = None
#                     ctx = getattr(m, "creation_context", None)
#                     if ctx and getattr(ctx, "created_at", None):
#                         try:
#                             created_on = ctx.created_at.astimezone(timezone.utc).date().isoformat()
#                         except Exception:
#                             created_on = None
#                         created_by = getattr(ctx, "created_by", None)

#                     nv = quote(f"{m.name}:{m.version}", safe=":")
#                     aml_ver_url = (
#                         f"https://ml.azure.com/model/{nv}/details"
#                         f"?wsid=/subscriptions/{sub_id}/resourceGroups/{rg}/workspaces/{wsn}{TENANT_QP}#overview"
#                     )
#                     last_mod = getattr(ctx, "last_modified_at", None) if ctx else None

#                     ver_meta = {
#                         "external_id": m.id,
#                         "name":        m.name,
#                         "version":     m.version,
#                         "description": m.description,
#                         "tags":        m.tags or {},
#                         "created_on":  created_on,
#                         "created_by":  created_by,
#                         "url_azureml": aml_ver_url,
#                         "last_modified": last_mod.astimezone(timezone.utc).isoformat() if last_mod else None,
#                         "modified_by":   getattr(ctx, "last_modified_by", None) if ctx else None,
#                         "path":         getattr(m, "path", None),
#                     }
#                     for k in ["model_card","approval_status","risk_assessment","bias_metrics","performance_metrics","explainability"]:
#                         v = (m.tags or {}).get(k)
#                         if v is not None:
#                             ver_meta[k] = v

#                     ver_art, ver_action = upsert_model_version(ver_meta, model_artifact_id=model_art.artifact_id)
#                     link_version_under_model(model_art, ver_art.artifact_id)
#                     print(f"VERSION {ver_action.upper():7} {m.name} v{m.version}")
#                     summary.append((wsn, m.name, str(m.version), ver_action))

#                     # datasets (attached to the version)
#                     job_name = infer_job_name_from_model(m)
#                     if not job_name:
#                         continue

#                     job = ml.jobs.get(job_name)
#                     inputs = getattr(job, "inputs", {}) or {}
#                     wsid = f"/subscriptions/{sub_id}/resourceGroups/{rg}/workspaces/{wsn}"

#                     d_ver = ver_art.get_definition()  # save once at end

#                     for key, inp in inputs.items():
#                         if not hasattr(inp, 'type') or inp.type not in [AssetTypes.URI_FILE, AssetTypes.URI_FOLDER, AssetTypes.MLTABLE]:
#                                 print(f"  SKIPPING input '{key}': Not a data asset (type: {getattr(inp, 'type', 'N/A')}).")
#                                 continue
#                         ref, da = resolve_input_to_data_asset(ml, inp)
#                         if not da:
#                             continue

#                         try:
#                             created_on = getattr(da, "creation_context", None)
#                             created_on = created_on.created_at.astimezone(timezone.utc).date().isoformat() if (created_on and created_on.created_at) else None
#                             created_by = getattr(da.creation_context, "created_by", None) if getattr(da, "creation_context", None) else None
#                             last_mod   = getattr(da, "creation_context", None)
#                             last_mod   = last_mod.last_modified_at.astimezone(timezone.utc).isoformat() if (last_mod and last_mod.last_modified_at) else None
#                             modified_by = getattr(da.creation_context, "last_modified_by", None) if getattr(da, "creation_context", None) else None
#                         except Exception:
#                             created_on = created_by = last_mod = modified_by = None

#                         data_url = (
#                             f"https://ml.azure.com/data/{quote(da.name)}/versions/{da.version}/details"
#                             f"?wsid={wsid}{TENANT_QP}"
#                         )

#                         ds_meta = {
#                             "external_id": da.id,
#                             "name":        da.name,
#                             "version":     da.version,
#                             "description": getattr(da, "description", None),
#                             "tags":        getattr(da, "tags", None) or {},
#                             "created_on":  created_on,
#                             "created_by":  created_by,
#                             "last_modified": last_mod,
#                             "modified_by":   modified_by,
#                             "path":        getattr(da, "path", None),
#                             "url_azureml": data_url,
#                         }

#                         ds_art, ds_action = upsert_dataset(ds_meta, model_version_artifact_id=ver_art.artifact_id)
#                         _set_reference(d_ver, F_MODEL_DATASETS, ds_art.artifact_id, list_field=True)
#                         print(f"  {ds_action.upper():7} DATASET {da.name}:{da.version} → linked to version {m.name}:{m.version}")

#                     d_ver.save()

#         # return a compact summary as the runnable’s output
#         return "Sync completed. Check the job log for per-item details."
