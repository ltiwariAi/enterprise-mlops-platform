"""
Synthetic Transaction Data Generator
Enterprise MLOps Platform

Generates 500,000 realistic credit card transactions for fraud detection.
All data is fully synthetic — no real individuals, accounts, or merchants.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import hashlib

np.random.seed(42)
random.seed(42)


# ============================================================
# MERCHANT DATABASE
# ============================================================

MERCHANTS = {
    "grocery": {
        "names": ["Fresh Market", "GreenGrocer", "Metro Foods", "Valley Superstore",
                   "CityMart", "Harvest Foods", "Corner Market", "QuickStop Grocery",
                   "Wholesome Foods", "Daily Basket", "Sunrise Market", "FoodVille"],
        "amount_range": (1.50, 85.00), "amount_median": 12.00,
        "freq_per_month": (12, 25), "mcc": "5411", "online_pct": 0.15,
    },
    "restaurant": {
        "names": ["QuickBite", "Burger Central", "Noodle House", "Taco Station",
                   "Sub Factory", "Chicken Shack", "Fish & Co", "Curry Corner",
                   "Wrap It Up", "Bowl & Roll", "Grill Master", "Pizza Palace"],
        "amount_range": (2.00, 45.00), "amount_median": 15.00,
        "freq_per_month": (6, 18), "mcc": "5812", "online_pct": 0.35,
    },
    "coffee": {
        "names": ["Bean Counter", "Morning Brew", "Espresso Lane", "The Roastery",
                   "Cuppa Joe", "Daily Grind", "Cafe Central", "Steam & Sip"],
        "amount_range": (2.50, 12.00), "amount_median": 5.50,
        "freq_per_month": (4, 15), "mcc": "5814", "online_pct": 0.0,
    },
    "transport_rail": {
        "names": ["CityRail", "Express Trains", "Metro Transit", "National Rail Co",
                   "Regional Express", "InterCity Lines"],
        "amount_range": (3.00, 55.00), "amount_median": 12.00,
        "freq_per_month": (3, 16), "mcc": "4112", "online_pct": 0.80,
    },
    "rideshare": {
        "names": ["RideNow", "GoTrip", "CityRide", "QuickCab", "ZipRide"],
        "amount_range": (3.00, 95.00), "amount_median": 10.00,
        "freq_per_month": (1, 10), "mcc": "4121", "online_pct": 1.0,
    },
    "retail_clothing": {
        "names": ["StyleHouse", "Urban Threads", "Classic Wear", "Trend Spot",
                   "ModaLife", "Apex Sports", "Fusion Fashion", "Denim Republic",
                   "StreetStyle", "Luxe Basics", "FitGear"],
        "amount_range": (10.00, 800.00), "amount_median": 45.00,
        "freq_per_month": (0, 4), "mcc": "5651", "online_pct": 0.30,
    },
    "retail_discount": {
        "names": ["ValueMart", "Discount Direct", "BargainBox", "SmartShop",
                   "Everything Plus", "PoundSaver"],
        "amount_range": (0.75, 25.00), "amount_median": 3.00,
        "freq_per_month": (1, 8), "mcc": "5331", "online_pct": 0.05,
    },
    "online_shopping": {
        "names": ["WebShop", "ClickBuy", "eMarket", "OnlineDeals", "NetStore",
                   "DigitalMart", "ShopDirect"],
        "amount_range": (3.00, 250.00), "amount_median": 15.00,
        "freq_per_month": (1, 6), "mcc": "5999", "online_pct": 1.0,
    },
    "subscription": {
        "names": ["StreamPlus", "CloudService", "TechSub", "MediaFlow",
                   "SoftwareCo", "AppService", "DigitalLife"],
        "amount_range": (4.99, 30.00), "amount_median": 10.49,
        "freq_per_month": (1, 4), "mcc": "5815", "online_pct": 1.0,
    },
    "hotel": {
        "names": ["City Inn", "Metro Hotel", "Comfort Suites", "Park Lodge",
                   "Garden Hotel", "Express Stay", "Airport Lodge", "Central Hotel"],
        "amount_range": (15.00, 1500.00), "amount_median": 120.00,
        "freq_per_month": (0, 2), "mcc": "7011", "online_pct": 0.70,
    },
    "telecom": {
        "names": ["MobileNet", "TeleCo", "ConnectPlus", "SignalOne"],
        "amount_range": (10.00, 60.00), "amount_median": 30.00,
        "freq_per_month": (0, 2), "mcc": "4812", "online_pct": 1.0,
    },
    "fuel": {
        "names": ["PetroStop", "FuelUp", "QuickGas", "CityFuel", "GoFuel"],
        "amount_range": (15.00, 90.00), "amount_median": 45.00,
        "freq_per_month": (0, 6), "mcc": "5541", "online_pct": 0.0,
    },
    "pharmacy": {
        "names": ["HealthMart", "CityPharmacy", "WellCare", "MediShop"],
        "amount_range": (2.00, 60.00), "amount_median": 12.00,
        "freq_per_month": (0, 3), "mcc": "5912", "online_pct": 0.20,
    },
    "entertainment": {
        "names": ["FunZone", "CinePlex", "GameWorld", "EventTix", "LiveArena"],
        "amount_range": (5.00, 150.00), "amount_median": 25.00,
        "freq_per_month": (0, 3), "mcc": "7832", "online_pct": 0.50,
    },
    "jewellery": {
        "names": ["GoldCraft", "Sparkle & Co", "Diamond Lane", "Silver Touch"],
        "amount_range": (20.00, 500.00), "amount_median": 80.00,
        "freq_per_month": (0, 1), "mcc": "5944", "online_pct": 0.30,
    },
}

# Cities with realistic geographic clustering
CITIES = {
    "UK": [
        {"city": "London", "state": "England", "country": "GB", "weight": 0.25},
        {"city": "Brighton", "state": "Sussex", "country": "GB", "weight": 0.15},
        {"city": "Manchester", "state": "England", "country": "GB", "weight": 0.08},
        {"city": "Birmingham", "state": "England", "country": "GB", "weight": 0.06},
        {"city": "Leeds", "state": "England", "country": "GB", "weight": 0.05},
        {"city": "Edinburgh", "state": "Scotland", "country": "GB", "weight": 0.04},
        {"city": "Bristol", "state": "England", "country": "GB", "weight": 0.04},
        {"city": "Reading", "state": "Berkshire", "country": "GB", "weight": 0.05},
        {"city": "Oxford", "state": "Oxfordshire", "country": "GB", "weight": 0.03},
        {"city": "Cambridge", "state": "Cambridgeshire", "country": "GB", "weight": 0.03},
    ],
    "US": [
        {"city": "New York", "state": "NY", "country": "US", "weight": 0.30},
        {"city": "San Francisco", "state": "CA", "country": "US", "weight": 0.15},
        {"city": "Chicago", "state": "IL", "country": "US", "weight": 0.10},
        {"city": "Los Angeles", "state": "CA", "country": "US", "weight": 0.10},
        {"city": "Boston", "state": "MA", "country": "US", "weight": 0.08},
        {"city": "Washington", "state": "DC", "country": "US", "weight": 0.07},
        {"city": "Miami", "state": "FL", "country": "US", "weight": 0.05},
        {"city": "Seattle", "state": "WA", "country": "US", "weight": 0.05},
        {"city": "Austin", "state": "TX", "country": "US", "weight": 0.05},
        {"city": "Denver", "state": "CO", "country": "US", "weight": 0.05},
    ],
    "IN": [
        {"city": "Bengaluru", "state": "Karnataka", "country": "IN", "weight": 0.40},
        {"city": "Mumbai", "state": "Maharashtra", "country": "IN", "weight": 0.25},
        {"city": "Delhi", "state": "Delhi", "country": "IN", "weight": 0.20},
        {"city": "Hyderabad", "state": "Telangana", "country": "IN", "weight": 0.15},
    ],
}


def generate_cardholder_id(n):
    return f"CH-{hashlib.md5(str(n).encode()).hexdigest()[:8].upper()}"


def generate_amount(category):
    cat = MERCHANTS[category]
    low, high = cat["amount_range"]
    median = cat["amount_median"]
    amount = np.random.lognormal(mean=np.log(median), sigma=0.6)
    amount = np.clip(amount, low, high)
    return round(amount, 2)


def generate_timestamp(base_date, hour_weights=None):
    if hour_weights is None:
        hour_weights = [0.01]*6 + [0.03, 0.05, 0.07, 0.08, 0.07, 0.06,
                        0.08, 0.07, 0.06, 0.05, 0.04, 0.05,
                        0.06, 0.07, 0.05, 0.03, 0.02, 0.01]
    hour_weights = np.array(hour_weights) / sum(hour_weights)
    hour = np.random.choice(24, p=hour_weights)
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    return base_date.replace(hour=hour, minute=minute, second=second)


def pick_city(profile_region, is_travel=False):
    if is_travel:
        region = random.choice(["UK", "US", "IN"])
    else:
        region = profile_region

    cities = CITIES[region]
    weights = [c["weight"] for c in cities]
    weights = np.array(weights) / sum(weights)
    city = np.random.choice(len(cities), p=weights)
    return cities[city]


# ============================================================
# CARDHOLDER PROFILES
# ============================================================

PROFILE_TYPES = {
    "commuter": {
        "weight": 0.30,
        "primary_categories": ["grocery", "coffee", "transport_rail", "restaurant", "subscription"],
        "monthly_spend": (400, 1500),
        "region": "UK",
        "travel_pct": 0.05,
    },
    "family": {
        "weight": 0.25,
        "primary_categories": ["grocery", "fuel", "restaurant", "retail_clothing", "pharmacy", "entertainment", "subscription"],
        "monthly_spend": (800, 3000),
        "region": "UK",
        "travel_pct": 0.03,
    },
    "young_professional": {
        "weight": 0.20,
        "primary_categories": ["coffee", "restaurant", "rideshare", "retail_clothing", "online_shopping", "subscription", "entertainment"],
        "monthly_spend": (500, 2000),
        "region": "UK",
        "travel_pct": 0.08,
    },
    "business_traveler": {
        "weight": 0.15,
        "primary_categories": ["hotel", "rideshare", "restaurant", "coffee", "transport_rail", "subscription"],
        "monthly_spend": (1000, 5000),
        "region": "UK",
        "travel_pct": 0.25,
    },
    "student": {
        "weight": 0.10,
        "primary_categories": ["grocery", "coffee", "restaurant", "retail_discount", "online_shopping", "subscription"],
        "monthly_spend": (150, 600),
        "region": "UK",
        "travel_pct": 0.02,
    },
}


# ============================================================
# FRAUD PATTERNS
# ============================================================

def inject_card_testing(transactions, cardholder_id, start_date):
    """Small rapid transactions at random merchants — testing stolen card."""
    fraud_txns = []
    num_tests = random.randint(3, 8)
    base_time = generate_timestamp(start_date)

    for i in range(num_tests):
        minutes_offset = random.randint(1, 15) * (i + 1)
        ts = base_time + timedelta(minutes=minutes_offset)
        cat = random.choice(list(MERCHANTS.keys()))
        merchant = random.choice(MERCHANTS[cat]["names"])
        city = pick_city("UK")

        fraud_txns.append({
            "transaction_id": f"TXN-{hashlib.md5(f'fraud_test_{cardholder_id}_{i}'.encode()).hexdigest()[:10].upper()}",
            "cardholder_id": cardholder_id,
            "timestamp": ts,
            "merchant_name": merchant,
            "merchant_category": cat,
            "mcc": MERCHANTS[cat]["mcc"],
            "merchant_city": city["city"],
            "merchant_state": city["state"],
            "merchant_country": city["country"],
            "amount": round(random.uniform(0.50, 5.00), 2),
            "currency": "GBP",
            "is_online": random.random() < 0.5,
            "is_recurring": False,
            "is_international": False,
            "is_fraud": 1,
            "fraud_type": "card_testing",
        })
    return fraud_txns


def inject_account_takeover(transactions, cardholder_id, start_date, profile):
    """Sudden spending spike in unusual categories."""
    fraud_txns = []
    unusual_cats = [c for c in MERCHANTS.keys() if c not in profile["primary_categories"]]
    num_txns = random.randint(3, 10)

    for i in range(num_txns):
        day_offset = random.randint(0, 2)
        ts = generate_timestamp(start_date + timedelta(days=day_offset))
        cat = random.choice(unusual_cats)
        merchant = random.choice(MERCHANTS[cat]["names"])
        city = pick_city("UK")

        fraud_txns.append({
            "transaction_id": f"TXN-{hashlib.md5(f'fraud_ato_{cardholder_id}_{i}'.encode()).hexdigest()[:10].upper()}",
            "cardholder_id": cardholder_id,
            "timestamp": ts,
            "merchant_name": merchant,
            "merchant_category": cat,
            "mcc": MERCHANTS[cat]["mcc"],
            "merchant_city": city["city"],
            "merchant_state": city["state"],
            "merchant_country": city["country"],
            "amount": round(random.uniform(100, 2000), 2),
            "currency": "GBP",
            "is_online": random.random() < 0.7,
            "is_recurring": False,
            "is_international": False,
            "is_fraud": 1,
            "fraud_type": "account_takeover",
        })
    return fraud_txns


def inject_geo_impossible(transactions, cardholder_id, start_date):
    """Two transactions in different countries within 1 hour."""
    fraud_txns = []
    ts1 = generate_timestamp(start_date)
    ts2 = ts1 + timedelta(minutes=random.randint(15, 45))

    city1 = pick_city("UK")
    city2 = pick_city("US")

    for i, (ts, city, country_code) in enumerate([(ts1, city1, "GBP"), (ts2, city2, "USD")]):
        cat = random.choice(["retail_clothing", "restaurant", "online_shopping"])
        merchant = random.choice(MERCHANTS[cat]["names"])

        fraud_txns.append({
            "transaction_id": f"TXN-{hashlib.md5(f'fraud_geo_{cardholder_id}_{i}'.encode()).hexdigest()[:10].upper()}",
            "cardholder_id": cardholder_id,
            "timestamp": ts,
            "merchant_name": merchant,
            "merchant_category": cat,
            "mcc": MERCHANTS[cat]["mcc"],
            "merchant_city": city["city"],
            "merchant_state": city["state"],
            "merchant_country": city["country"],
            "amount": round(random.uniform(50, 500), 2),
            "currency": country_code,
            "is_online": False,
            "is_recurring": False,
            "is_international": city["country"] != "GB",
            "is_fraud": 1,
            "fraud_type": "geo_impossible",
        })
    return fraud_txns


def inject_bustout(transactions, cardholder_id, start_date, profile):
    """Gradual spending increase then max-out."""
    fraud_txns = []
    normal_max = profile["monthly_spend"][1] / 30

    for day in range(14):
        multiplier = 1 + (day * 0.5)
        num_daily = random.randint(1, 3 + day // 3)
        ts_base = start_date + timedelta(days=day)

        for i in range(num_daily):
            ts = generate_timestamp(ts_base)
            cat = random.choice(list(MERCHANTS.keys()))
            merchant = random.choice(MERCHANTS[cat]["names"])
            city = pick_city("UK")
            amount = round(normal_max * multiplier * random.uniform(0.5, 1.5), 2)

            fraud_txns.append({
                "transaction_id": f"TXN-{hashlib.md5(f'fraud_bust_{cardholder_id}_{day}_{i}'.encode()).hexdigest()[:10].upper()}",
                "cardholder_id": cardholder_id,
                "timestamp": ts,
                "merchant_name": merchant,
                "merchant_category": cat,
                "mcc": MERCHANTS[cat]["mcc"],
                "merchant_city": city["city"],
                "merchant_state": city["state"],
                "merchant_country": city["country"],
                "amount": min(amount, 5000),
                "currency": "GBP",
                "is_online": random.random() < 0.4,
                "is_recurring": False,
                "is_international": False,
                "is_fraud": 1,
                "fraud_type": "bust_out",
            })
    return fraud_txns


def inject_cnp_fraud(transactions, cardholder_id, start_date):
    """Card-not-present fraud — online purchases in unusual categories."""
    fraud_txns = []
    num_txns = random.randint(3, 7)

    for i in range(num_txns):
        hours_offset = random.randint(0, 48)
        ts = start_date + timedelta(hours=hours_offset)
        cat = random.choice(["online_shopping", "retail_clothing", "entertainment", "jewellery"])
        merchant = random.choice(MERCHANTS[cat]["names"])
        city = pick_city("US")

        fraud_txns.append({
            "transaction_id": f"TXN-{hashlib.md5(f'fraud_cnp_{cardholder_id}_{i}'.encode()).hexdigest()[:10].upper()}",
            "cardholder_id": cardholder_id,
            "timestamp": ts,
            "merchant_name": merchant,
            "merchant_category": cat,
            "mcc": MERCHANTS[cat]["mcc"],
            "merchant_city": city["city"],
            "merchant_state": city["state"],
            "merchant_country": city["country"],
            "amount": round(random.uniform(50, 800), 2),
            "currency": "USD",
            "is_online": True,
            "is_recurring": False,
            "is_international": True,
            "is_fraud": 1,
            "fraud_type": "cnp_fraud",
        })
    return fraud_txns


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_legitimate_transactions(cardholder_id, profile_type, start_date, end_date):
    """Generate realistic legitimate transactions for one cardholder."""
    profile = PROFILE_TYPES[profile_type]
    transactions = []
    monthly_budget = random.uniform(*profile["monthly_spend"])
    
    current_date = start_date
    while current_date < end_date:
        month_end = min(current_date + timedelta(days=30), end_date)
        month_spend = 0

        for category in profile["primary_categories"]:
            cat_info = MERCHANTS[category]
            num_txns = random.randint(*cat_info["freq_per_month"])

            for _ in range(num_txns):
                if month_spend > monthly_budget * 1.2:
                    break

                day_offset = random.randint(0, min(29, (month_end - current_date).days))
                tx_date = current_date + timedelta(days=day_offset)
                ts = generate_timestamp(tx_date)

                merchant = random.choice(cat_info["names"])
                is_travel = random.random() < profile["travel_pct"]
                is_international = is_travel and random.random() < 0.5

                if is_international:
                    city = pick_city(profile["region"], is_travel=True)
                    currency = {"GB": "GBP", "US": "USD", "IN": "INR"}.get(city["country"], "GBP")
                else:
                    city = pick_city(profile["region"])
                    currency = "GBP"

                amount = generate_amount(category)
                is_online = random.random() < cat_info["online_pct"]
                is_recurring = category == "subscription" and random.random() < 0.8

                transactions.append({
                    "transaction_id": f"TXN-{hashlib.md5(f'{cardholder_id}_{ts}_{merchant}'.encode()).hexdigest()[:10].upper()}",
                    "cardholder_id": cardholder_id,
                    "timestamp": ts,
                    "merchant_name": merchant,
                    "merchant_category": category,
                    "mcc": cat_info["mcc"],
                    "merchant_city": city["city"],
                    "merchant_state": city["state"],
                    "merchant_country": city["country"],
                    "amount": amount,
                    "currency": currency,
                    "is_online": is_online,
                    "is_recurring": is_recurring,
                    "is_international": is_international,
                    "is_fraud": 0,
                    "fraud_type": "none",
                })
                month_spend += amount

        # Add occasional random category transactions
        for _ in range(random.randint(0, 5)):
            random_cat = random.choice(list(MERCHANTS.keys()))
            if random_cat not in profile["primary_categories"]:
                day_offset = random.randint(0, min(29, (month_end - current_date).days))
                tx_date = current_date + timedelta(days=day_offset)
                ts = generate_timestamp(tx_date)
                merchant = random.choice(MERCHANTS[random_cat]["names"])
                city = pick_city(profile["region"])
                amount = generate_amount(random_cat)

                transactions.append({
                    "transaction_id": f"TXN-{hashlib.md5(f'{cardholder_id}_{ts}_{merchant}_rnd'.encode()).hexdigest()[:10].upper()}",
                    "cardholder_id": cardholder_id,
                    "timestamp": ts,
                    "merchant_name": merchant,
                    "merchant_category": random_cat,
                    "mcc": MERCHANTS[random_cat]["mcc"],
                    "merchant_city": city["city"],
                    "merchant_state": city["state"],
                    "merchant_country": city["country"],
                    "amount": amount,
                    "currency": "GBP",
                    "is_online": random.random() < MERCHANTS[random_cat]["online_pct"],
                    "is_recurring": False,
                    "is_international": False,
                    "is_fraud": 0,
                    "fraud_type": "none",
                })

        current_date = month_end

    return transactions


def generate_dataset(num_cardholders=5000, months=6, target_rows=500000, fraud_rate=0.002):
    """Generate the full synthetic dataset."""
    print(f"Generating synthetic transaction data...")
    print(f"  Cardholders: {num_cardholders}")
    print(f"  Timespan: {months} months")
    print(f"  Target rows: ~{target_rows:,}")
    print(f"  Target fraud rate: {fraud_rate*100:.1f}%")

    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=months * 30)

    all_transactions = []
    profile_types = list(PROFILE_TYPES.keys())
    profile_weights = [PROFILE_TYPES[p]["weight"] for p in profile_types]

    # Generate legitimate transactions
    for i in range(num_cardholders):
        if i % 500 == 0:
            print(f"  Processing cardholder {i+1}/{num_cardholders}...")

        ch_id = generate_cardholder_id(i)
        profile_type = np.random.choice(profile_types, p=profile_weights)
        txns = generate_legitimate_transactions(ch_id, profile_type, start_date, end_date)
        all_transactions.extend(txns)

    print(f"  Legitimate transactions: {len(all_transactions):,}")

    # Inject fraud
    num_fraud_cardholders = int(num_cardholders * fraud_rate * 50)
    fraud_types = [inject_card_testing, inject_account_takeover, inject_geo_impossible, inject_bustout, inject_cnp_fraud]
    fraud_transactions = []

    for i in range(num_fraud_cardholders):
        ch_id = generate_cardholder_id(random.randint(0, num_cardholders - 1))
        fraud_date = start_date + timedelta(days=random.randint(30, months * 30 - 14))
        fraud_func = random.choice(fraud_types)
        profile_type = np.random.choice(profile_types, p=profile_weights)
        profile = PROFILE_TYPES[profile_type]

        if fraud_func in [inject_account_takeover, inject_bustout]:
            fraud_txns = fraud_func([], ch_id, fraud_date, profile)
        else:
            fraud_txns = fraud_func([], ch_id, fraud_date)

        fraud_transactions.extend(fraud_txns)

    all_transactions.extend(fraud_transactions)
    print(f"  Fraud transactions: {len(fraud_transactions):,}")

    # Build DataFrame
    df = pd.DataFrame(all_transactions)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Trim or pad to target size
    if len(df) > target_rows:
        df = df.sample(n=target_rows, random_state=42).sort_values("timestamp").reset_index(drop=True)

    # Final stats
    fraud_count = df["is_fraud"].sum()
    total = len(df)
    print(f"\n  DATASET GENERATED")
    print(f"  Total transactions: {total:,}")
    print(f"  Legitimate: {total - fraud_count:,} ({(total - fraud_count)/total*100:.2f}%)")
    print(f"  Fraudulent: {fraud_count:,} ({fraud_count/total*100:.3f}%)")
    print(f"\n  Fraud breakdown:")
    for ft in df[df["is_fraud"] == 1]["fraud_type"].value_counts().items():
        print(f"    {ft[0]}: {ft[1]:,}")
    print(f"\n  Categories:")
    for cat in df["merchant_category"].value_counts().head(10).items():
        print(f"    {cat[0]}: {cat[1]:,}")
    print(f"\n  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


if __name__ == "__main__":
    df = generate_dataset(num_cardholders=5000, months=6, target_rows=500000, fraud_rate=0.002)
    
    output_path = "data/raw/synthetic_transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Saved to {output_path}")
    print(f"  File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB (in memory)")
    
    print(f"\n  Sample transactions:")
    print(df.head(10).to_string(index=False))
