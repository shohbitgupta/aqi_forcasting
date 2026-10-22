"""Cities used in the project, with coordinates for downloading weather data.

The keys must match the `area` names used in the CPCB AQI bulletin file.
If prepare_aqi.py reports a city as "not found", it prints the closest
matching names from the file so you can fix the spelling here.
"""

CITIES = {
    # Delhi NCR
    "Delhi": (28.6139, 77.2090),
    "Noida": (28.5355, 77.3910),
    "Greater Noida": (28.4744, 77.5040),
    "Ghaziabad": (28.6692, 77.4538),
    "Gurugram": (28.4595, 77.0266),
    "Faridabad": (28.4089, 77.3178),
    # Indo-Gangetic Plain
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Patna": (25.5941, 85.1376),
    # Other metros (cleaner air, useful contrast for the model)
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
}

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
