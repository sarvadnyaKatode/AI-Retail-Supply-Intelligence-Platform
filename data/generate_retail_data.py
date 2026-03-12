import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# -----------------------------------
# CONFIGURATION
# -----------------------------------

start_date = datetime(2022, 1, 1)
days = 1095   # 3 years of data

stores = [
    "Mumbai","Pune","Nagpur","Nashik","Aurangabad",
    "Thane","Kolhapur","Solapur","Amravati","Akola"
]

store_types = {
    "Mumbai":"supermarket",
    "Pune":"supermarket",
    "Nagpur":"supermarket",
    "Nashik":"convenience",
    "Aurangabad":"convenience",
    "Thane":"convenience",
    "Kolhapur":"mini_store",
    "Solapur":"mini_store",
    "Amravati":"mini_store",
    "Akola":"mini_store"
}

store_multiplier = {
    "supermarket":1.3,
    "convenience":1.1,
    "mini_store":0.9
}

products = [
    "milk","bread","rice","onion","potato",
    "eggs","sugar","oil","tomato","salt"
]

base_prices = {
    "milk":50,"bread":30,"rice":60,"onion":40,"potato":35,
    "eggs":70,"sugar":45,"oil":120,"tomato":30,"salt":20
}

festival_dates = [
    "2022-10-24","2023-11-12","2024-11-01",  # Diwali
    "2022-03-18","2023-03-08","2024-03-25",  # Holi
    "2022-12-25","2023-12-25","2024-12-25"   # Christmas
]

festival_dates = [pd.to_datetime(d) for d in festival_dates]

rows = []

# -----------------------------------
# DATA GENERATION
# -----------------------------------

for store in stores:

    store_type = store_types[store]
    demand_multiplier = store_multiplier[store_type]

    for product in products:

        base_demand = np.random.randint(20,60)

        for day in range(days):

            date = start_date + timedelta(days=day)
            weekday = date.weekday()

            # -------------------------
            # TREND
            # -------------------------

            trend = 1 + (day/days)*0.3

            # -------------------------
            # WEEKEND EFFECT
            # -------------------------

            weekend_spike = 1.25 if weekday >=5 else 1

            # -------------------------
            # SEASONALITY
            # -------------------------

            seasonal = 1 + 0.15*np.sin(day/30)

            # -------------------------
            # FESTIVAL BOOST
            # -------------------------

            festival = 1 if date in festival_dates else 0
            festival_boost = 1.5 if festival else 1

            # -------------------------
            # WEATHER
            # -------------------------

            temperature = np.random.uniform(18,35)
            rainfall_mm = np.random.uniform(0,20)

            weather_factor = 1.1 if rainfall_mm > 10 else 1

            # -------------------------
            # PROMOTIONS
            # -------------------------

            promotion = np.random.choice([0,1], p=[0.85,0.15])

            if promotion:
                discount_pct = np.random.randint(5,30)
            else:
                discount_pct = 0

            promotion_boost = 1 + discount_pct/100

            # -------------------------
            # DEMAND CALCULATION
            # -------------------------

            demand = (
                base_demand *
                trend *
                weekend_spike *
                seasonal *
                festival_boost *
                weather_factor *
                promotion_boost *
                demand_multiplier
            )

            sales = max(0, int(np.random.normal(demand,5)))

            # -------------------------
            # DYNAMIC PRICING
            # -------------------------

            base_price = base_prices[product]

            price_variation = np.random.uniform(-0.1,0.15)
            price = base_price * (1 + price_variation)

            competitor_price = price * np.random.uniform(0.9,1.1)

            # -------------------------
            # INVENTORY
            # -------------------------

            inventory = sales + np.random.randint(10,80)

            stockout = 1 if sales > inventory else 0

            # -------------------------
            # SUPPLIER LEAD TIME
            # -------------------------

            lead_time = np.random.randint(1,5)

            rows.append([
                date,
                store,
                store_type,
                product,
                round(price,2),
                round(competitor_price,2),
                promotion,
                discount_pct,
                sales,
                inventory,
                stockout,
                temperature,
                rainfall_mm,
                festival,
                weekday,
                1 if weekday >=5 else 0,
                lead_time
            ])

# -----------------------------------
# DATAFRAME
# -----------------------------------

df = pd.DataFrame(rows, columns=[
    "date",
    "store",
    "store_type",
    "product",
    "price",
    "competitor_price",
    "promotion",
    "discount_pct",
    "sales",
    "inventory",
    "stockout",
    "temperature",
    "rainfall_mm",
    "festival",
    "day_of_week",
    "is_weekend",
    "supplier_lead_time"
])

df.to_csv("data/retail_demand_dataset.csv", index=False)

print("Dataset generated successfully")
print("Total rows:", len(df))