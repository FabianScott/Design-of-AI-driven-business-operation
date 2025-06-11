import numpy as np
# KHvm
transport_modes = {
    1: "Passenger car",
    2: "Train",
    3: "Bus",
    4: "Tram",
    5: "Metro",
    6: "Speed pedelec",
    7: "Electric bicycle",
    8: "Non-electric bicycle",
    9: "On foot",
    10: "Touring car",
    11: "Delivery van",
    12: "Truck",
    13: "Camper",
    14: "Taxi/Minibus",
    15: "Agricultural vehicle",
    16: "Motorcycle",
    17: "Moped",
    18: "Light moped",
    19: "Mobility aid with motor",
    20: "Mobility aid without motor",
    21: "Skates/inline skates/kick scooter",
    22: "Boat",
    23: "Other with motor",
    24: "Other without motor"
}

# KMotiefV
# KHvm
trip_motives = {
    1: "Commuting (to and from work)",
    2: "Business visit (at work location)",
    3: "Occupational (work-related)",
    4: "Picking up/dropping off people",
    5: "Picking up/dropping off goods",
    6: "Education/course attendance",
    7: "Shopping/grocery shopping",
    8: "Visiting/staying over",
    9: "Touring/hiking",
    10: "Sports/hobby",
    11: "Other leisure activity",
    12: "Services/personal care",
    13: "Other motive"
}


level_mapping_suffix = {
    0: "PC",
    1: "Gem",
    2: "Prov"
}

province_codes = {
    0: "No departure point in the Netherlands",
    1: "Groningen",
    2: "Friesland",
    3: "Drenthe",
    4: "Overijssel",
    5: "Flevoland",
    6: "Gelderland",
    7: "Utrecht",
    8: "North Holland",
    9: "South Holland",
    10: "Zeeland",
    11: "North Brabant",
    12: "Limburg",
    99: "Unknown"
}


purpose_col_dict = {
    1: "To home",
    2: "Work",
    3: "Business visit (work-related)",
    4: "Professional",
    5: "Pick up/drop off people",
    6: "Pick up/drop off goods",
    7: "Education/course",
    8: "Shopping/groceries",
    9: "Visit/stay over",
    10: "Touring/walking",
    11: "Sports/hobby",
    12: "Other leisure activity",
    13: "Services/personal care",
    14: "Other purpose"
}

motive_col_dict = {
    1: "Commute to/from work",
    2: "Business visit (work-related)",
    3: "Professional",
    4: "Pick up/drop off people",
    5: "Pick up/drop off goods",
    6: "Education/course",
    7: "Shopping/groceries",
    8: "Visit/stay over",
    9: "Touring/walking",
    10: "Sports/hobby",
    11: "Other leisure activity",
    12: "Services/personal care",
    13: "Other motive"
}

def AfstV_to_KAfstV(meters):
    km = meters / 1000
    # Define the upper bounds for the ranges (excluding last open-ended)
    bounds = np.array([0.5, 1.0, 2.5, 3.7, 5.0, 7.5, 10, 15, 20, 30, 40, 50, 75, 100])
    # Find the first index where km is less than the bound
    index = np.searchsorted(bounds, km, side='right')
    
    # Range comments:
    # 1: 0.1–0.5 km
    # 2: 0.5–1.0 km
    # 3: 1.0–2.5 km
    # 4: 2.5–3.7 km
    # 5: 3.7–5.0 km
    # 6: 5.0–7.5 km
    # 7: 7.5–10 km
    # 8: 10–15 km
    # 9: 15–20 km
    #10: 20–30 km
    #11: 30–40 km
    #12: 40–50 km
    #13: 50–75 km
    #14: 75–100 km
    #15: 100 km or more

    return index + 1 if index < 14 else 15