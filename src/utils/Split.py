# import pandas as pd
# file_path = r"D:\ada\archive\symbols_valid_meta.csv" # Load the CSV file
# data = pd.read_csv(file_path)

# etfs = data[data['ETF'] == 'Y']  
# stocks = data[data['ETF'] == 'N']

# # Save the separated data into new CSV files
# etfs.to_csv("etfs.csv", index=False)
# stocks.to_csv("stocks.csv", index=False)

# # Check
# print("ETFs and stocks have been separated and saved to 'etfs.csv' and 'stocks.csv'.")



# # Additional categorization of ETFs and based on keywords in their names
# file_path = r"d:\ada\stocks.csv"
# data = pd.read_csv(file_path)

# categories = {
#     "Energy": [
#         "energy", "oil", "gas", "o&g", "upstream", "midstream", "downstream",
#         "refining", "lng", "pipeline", "drilling", "oilfield services", "coal",
#         "consumable fuels"
#     ],
#     "Materials": [
#         "materials", "chemicals", "commodity chemicals", "specialty chemicals",
#         "construction materials", "cement", "glass", "containers", "packaging",
#         "paper", "forest products", "metals", "mining", "steel", "aluminum",
#         "copper", "gold", "silver", "precious metals"
#     ],
#     "Industrials": [
#         "industrial", "capital goods", "aerospace", "defense", "building products",
#         "construction & engineering", "electrical equipment", "machinery",
#         "conglomerates", "transportation", "airlines", "air freight", "logistics",
#         "marine", "rail", "road & rail", "professional services"
#     ],
#     "Consumer Discretionary": [
#         "discretionary", "auto", "automobile", "auto parts", "durables",
#         "leisure", "entertainment products", "diversified retail", "specialty retail",
#         "e-commerce retail", "online retail", "apparel", "luxury", "hotels",
#         "restaurants", "gaming (casinos)", "travel", "media retail"
#     ],
#     "Consumer Staples": [
#         "staples", "food", "beverage", "tobacco", "household products",
#         "personal care", "grocery", "supermarket", "packaged foods",
#         "distillers", "brewers", "soft drinks"
#     ],
#     "Health Care": [
#         "health care", "healthcare", "biotech", "pharma", "pharmaceuticals",
#         "life sciences", "genomics", "medical devices", "medical equipment",
#         "diagnostics", "therapeutics", "managed care", "hospital", "providers"
#     ],
#     "Financials": [
#         "financials", "bank", "banks", "insurance", "brokerage", "investment",
#         "asset management", "capital markets", "wealth", "mortgage", "loan",
#         "credit", "fintech", "consumer finance", "exchange", "market infrastructure"
#     ],
#     "Information Technology": [
#         "information technology", "technology", "tech", "semiconductor",
#         "chips", "hardware", "software", "saas", "cloud", "it services",
#         "computing", "robotics", "electronics", "cybersecurity", "enterprise software"
#     ],
#     "Communication Services": [
#         "communication services", "telecom", "wireless", "broadband", "5g",
#         "media", "broadcasting", "publishing", "advertising", "digital ads",
#         "streaming", "social media", "interactive media", "internet platforms",
#         "gaming (online)", "search", "messaging"
#     ],
#     "Utilities": [
#         "utilities", "electric utilities", "gas utilities", "water utilities",
#         "independent power", "renewable electricity", "power generation",
#         "grid", "transmission", "distribution"
#     ],
#     "Real Estate": [
#         "real estate", "reit", "property", "commercial", "residential",
#         "leasing", "development", "land", "mortgage reit"
#     ]
# }


# category_dfs = {category: pd.DataFrame() for category in categories}

# # 关键词分类
# for index, row in data.iterrows():
#     added = False
#     security_name = row["Security Name"].lower()
#     for category, keywords in categories.items():
#         if any(keyword in security_name for keyword in keywords):
#             category_dfs[category] = pd.concat([category_dfs[category], pd.DataFrame([row])])
#             added = True
#             break
#     if not added:
#         # "Other"
#         if "Other" not in category_dfs:
#             category_dfs["Other"] = pd.DataFrame()
#         category_dfs["Other"] = pd.concat([category_dfs["Other"], pd.DataFrame([row])])

# output_dir = r"d:\ada\stocks_categories"
# import os
# os.makedirs(output_dir, exist_ok=True)

# for category, df in category_dfs.items():
#     output_file = os.path.join(output_dir, f"{category.replace(' ', '_').lower()}.csv")
#     df.to_csv(output_file, index=False)

# print(f"分类完成！所有文件已保存到 {output_dir}")