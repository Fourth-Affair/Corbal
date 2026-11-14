from fastapi import FastAPI
from fastapi import HTTPException
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import UUID

app = FastAPI(title="Junc API", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/hello")
async def hello() -> dict[str, str]:
    return {"message": "Hello from FastAPI"}


@app.get("/api/plans/{plan_id}")
def get_plan(plan_id: UUID) -> dict:
    """Return a plan and its linked Storage objects by plan UUID."""
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_key:
        raise HTTPException(status_code=503, detail="Supabase backend credentials are not configured")

    query = urlencode({"id": f"eq.{plan_id}", "select": "id,created_at,metadata,plan_images(*)"})
    request = Request(
        f"{supabase_url.rstrip('/')}/rest/v1/plans?{query}",
        headers={"apikey": service_key, "Authorization": f"Bearer {service_key}"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            plans = json.loads(response.read())
    except HTTPError as error:
        raise HTTPException(status_code=502, detail="Supabase plan lookup failed") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="Could not reach Supabase") from error

    if not plans:
        raise HTTPException(status_code=404, detail="Plan not found")
    return plans[0]
