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
output_dir = "/umbc/rs/pi_halem/users/yashd1/ALL_ERA5/tcc"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "era5_total_cloud_cover_2022_jan_feb_6hourly.nc"
)

dataset = "reanalysis-era5-single-levels"

request = {
    "product_type": ["reanalysis"],
    "variable": ["total_cloud_cover"],
    "year": ["2022"],
    "month": ["01", "02", "03"],
    "day": [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12",
        "13", "14", "15", "16", "17", "18",
        "19", "20", "21", "22", "23", "24",
        "25", "26", "27", "28", "29", "30",
        "31"
    ],
    "time": ["00:00", "06:00", "12:00", "18:00"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client = cdsapi.Client(
    url=CDS_URL,
    key=CDS_KEY
)

client.retrieve(dataset, request).download(output_file)

print(f"Downloaded to: {output_file}")
