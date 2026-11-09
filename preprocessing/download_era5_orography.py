import os
import cdsapi

# ==========================
# Hardcoded CDS credentials
# ==========================
CDS_URL = "https://cds.climate.copernicus.eu/api"
CDS_KEY = os.environ["CDSAPI_KEY"]  # REDACTED for release: personal CDS API key removed

# ==========================
# Output path
# ==========================
output_dir = "umbc/rs/pi_halem/users/yashd1/ALL_ERA5/terrain"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "era5_land_geopotential_orography.nc"
)

# ==========================
# CDS request
# ==========================
dataset = "reanalysis-era5-land"

request = {
    "variable": ["geopotential"],
    "download_format": "unarchived"
}

client = cdsapi.Client(
    url=CDS_URL,
    key=CDS_KEY
)

client.retrieve(dataset, request).download(output_file)

print(f"Downloaded to: {output_file}")
