# Frost API — Reference Notes

API docs: https://frost.met.no/api.html

## Authentication

- HTTP Basic Auth: client ID as username, empty password
- Client ID stored in `.env` as `FROST_CLIENT_ID`

## Endpoints

| Endpoint | URL | Purpose |
|----------|-----|---------|
| Observations | `/observations/v0.jsonld` | Historical weather data |
| Available time series | `/observations/availableTimeSeries/v0.jsonld` | Check what data exists for a station |
| Elements | `/elements/v0.jsonld` | List available weather elements |
| Sources | `/sources/v0.jsonld` | List weather stations |
| Records | `/records/v0.jsonld` | Climate records |

## HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Invalid parameter value or malformed request |
| 401 | Unauthorized client ID |
| 404 | No data found for the query |
| 500 | Internal server error |

## Key Elements for Electricity Price Forecasting

| Element ID | Description | Unit |
|-----------|-------------|------|
| `air_temperature` | Air temperature | degC |
| `wind_speed` | Wind speed | m/s |
| `sum(precipitation_amount PT1H)` | Hourly precipitation | mm |
| `surface_snow_thickness` | Snow depth | cm |
| `cloud_area_fraction` | Cloud cover | % |
| `relative_humidity` | Relative humidity | % |

## Weather Stations per Bidding Zone

| Zone | Station | Frost ID | Coordinates |
|------|---------|----------|-------------|
| NO1 (Oslo) | Oslo - Blindern | SN18700 | 59.9423, 10.72 |
| NO2 (Kristiansand) | Kristiansand - Kjevik | SN39040 | 58.20, 8.08 |
| NO3 (Trondheim) | Trondheim - Voll | SN68860 | 63.41, 10.46 |
| NO4 (Tromsø) | Tromsø | SN90450 | 69.65, 18.94 |
| NO5 (Bergen) | Bergen - Florida | SN50540 | 60.38, 5.33 |

## Pagination

Frost API uses offset-based pagination:
- `itemsPerPage`: max items per response (default varies)
- `offset`: zero-based index for pagination
- `nextLink`: URL for next page of results
- Follow `nextLink` until it's absent to get all data

## Example Observation Response Structure

```json
{
  "sourceId": "SN18700:0",
  "referenceTime": "2024-01-01T00:00:00.000Z",
  "observations": [
    {
      "elementId": "air_temperature",
      "value": -3.2,
      "unit": "degC",
      "qualityCode": 0
    }
  ]
}
```

## Norwegian Counties (Fylker)

Akershus, Oslo, Vestland, Rogaland, Trøndelag, Innlandet, Agder,
Østfold, Møre og Romsdal, Buskerud, Vestfold, Nordland, Telemark,
Troms, Finnmark

## Quality Codes

| Code | Meaning |
|------|---------|
| 0 | OK — data has passed all quality control |
| 1 | Suspect — data has failed some quality checks |
| 2 | Incorrect — data is known to be wrong |

See: https://frost.met.no/api.html#!/observations/availableQualityCodes