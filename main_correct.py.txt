from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import traceback
import json
import logging
import re
from rapidfuzz import fuzz, process as rapidfuzz_process
from collections import defaultdict
import time
from flask import Flask, request, jsonify
from collections import defaultdict
from functools import lru_cache
from utils import (
    CATEGORY_CANONICAL,
    TWOWHEELER_AUTOSUGGEST,
    BUSINESS_SYNONYMS,
    CATEGORY_HARDCODED_CHIPS,
    APPLE_TERMS,
    FILTER_ATTRIBUTE_EXCLUSIONS,
    SCOOTER_SYNONYMS,
    BUSINESS_AUTOSUGGEST,
)



# =================== CONFIGS ===========================
PRODUCT_INDEX_NAME = "bajajmall_new_data_index_2106"
CATEGORY_INDEX_NAME = "bajajmall_new_data_category_index_2106"
AUTOSUGGEST_INDEX_NAME = "bajajmall_new_data_autosuggest_index_2106"
IMAGE_DOMAIN = "https://mc.bajajfinserv.in/media/catalog/product"


logging.basicConfig(filename="api.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = Flask(__name__)
es = Elasticsearch("http://localhost:9200")


def is_apple_only_query(query):
    query_n = normalize(query)
    tokens = [t for t in re.split(r'[\s,]+', query_n) if t]
    NON_APPLE_NOISE = set(["kit", "and", "pro", "air", "mini", "plus", "blynk"])
    return all(
        t in APPLE_TERMS or t in NON_APPLE_NOISE for t in tokens
    ) and any(t in APPLE_TERMS for t in tokens)







# Map all categories to lowercase for lookup
CATEGORY_CANONICAL_LOWER = {k.lower(): v for k, v in CATEGORY_CANONICAL.items()}
CATEGORY_CANONICAL_LOOKUP = {v.lower(): v for k, v in CATEGORY_CANONICAL.items()}
# Add synonym support:
ALL_SYNONYM_TO_CANONICAL = {}
for canon, syns in BUSINESS_SYNONYMS.items():
    for s in syns:
        ALL_SYNONYM_TO_CANONICAL[s.lower()] = canon.lower()
for k in CATEGORY_CANONICAL.keys():
    ALL_SYNONYM_TO_CANONICAL[k.lower()] = k.lower()
    ALL_SYNONYM_TO_CANONICAL[CATEGORY_CANONICAL[k].lower()] = k.lower()

# ============= NORMALIZATION/HELPERS ==============
def normalize(text):
    if not text: return ""
    t = re.sub(r"[^a-zA-Z0-9\s&-]", "", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build_correction_pool():
    pool = set()
    for k, v in CATEGORY_CANONICAL.items():
        pool.add(normalize(k))
        pool.add(normalize(v))
        pool.add(normalize(k.replace("-", "")))
        pool.add(normalize(k.replace(" ", "")))
    for syns in BUSINESS_SYNONYMS.values():
        for s in syns:
            pool.add(normalize(s))
            pool.add(normalize(s.replace("-", "")))
            pool.add(normalize(s.replace(" ", "")))
    return sorted({p for p in pool if p})

CORRECTION_POOL = build_correction_pool()

def correct_query(user_query):
    user_query_n = normalize(user_query)
    words = user_query_n.split()
    match, score, _ = rapidfuzz_process.extractOne(
        user_query_n, CORRECTION_POOL, scorer=fuzz.ratio, score_cutoff=78
    ) or (None, 0, None)
    if match and score >= 78:
        return match
    new_words = []
    for w in words:
        m, sc, _ = rapidfuzz_process.extractOne(
            w, CORRECTION_POOL, scorer=fuzz.ratio, score_cutoff=75
        ) or (w, 0, None)
        new_words.append(m if sc >= 75 else w)
    corrected_phrase = " ".join(new_words)
    if corrected_phrase != user_query_n:
        return corrected_phrase
    return user_query_n

import re
import re

def parse_price_from_query(query):
   
    q = query.lower()
    cleaned_query = q
    filters = {}

    # --------- 1. Handle explicit EMI mention ---------
    emi_patterns = [
        # e.g., 'emi under 2000', 'monthly under 2500', 'installment below 1800'
        (r'(lowest )?(emi|installment|monthly|per month)[^\d]{0,10}(under|below|less than|upto|<=)\s*([0-9]{3,8})',
         lambda m: ('lowest_emi', {'lte': int(m.group(4))})),
        (r'(lowest )?(emi|installment|monthly|per month)[^\d]{0,10}(above|over|greater than|more than|>=)\s*([0-9]{3,8})',
         lambda m: ('lowest_emi', {'gte': int(m.group(4))})),
        # e.g., 'emi from 1200 to 1500', 'emi 1200-1500'
        (r'(lowest )?(emi|installment|monthly|per month)[^\d]{0,10}(from|between|range)?\s*([0-9]{3,8})\s*(to|and|-)\s*([0-9]{3,8})',
         lambda m: ('lowest_emi', {'gte': min(int(m.group(4)), int(m.group(6))), 'lte': max(int(m.group(4)), int(m.group(6)))}))
    ]
    for pat, rng_fn in emi_patterns:
        for match in re.finditer(pat, cleaned_query):
            key, val = rng_fn(match)
            filters[key] = val
            cleaned_query = cleaned_query.replace(match.group(0), '').strip()

    # --------- 2. Handle price (MOP/offer_price) keywords ---------
    price_patterns = [
        (r'(price|cost|offer price|rate|mop)[^\d]{0,10}(under|below|less than|upto|<=)\s*([0-9]{3,8})',
         lambda m: ('mop', {'lte': int(m.group(3))})),
        (r'(price|cost|offer price|rate|mop)[^\d]{0,10}(above|over|greater than|more than|>=)\s*([0-9]{3,8})',
         lambda m: ('mop', {'gte': int(m.group(3))})),
        (r'(price|cost|offer price|rate|mop)[^\d]{0,10}(from|between|range)?\s*([0-9]{3,8})\s*(to|and|-)\s*([0-9]{3,8})',
         lambda m: ('mop', {'gte': min(int(m.group(4)), int(m.group(6))), 'lte': max(int(m.group(4)), int(m.group(6)))}))
    ]
    for pat, rng_fn in price_patterns:
        for match in re.finditer(pat, cleaned_query):
            key, val = rng_fn(match)
            filters[key] = val
            cleaned_query = cleaned_query.replace(match.group(0), '').strip()

    # --------- 3. "Under 25000 emi" (emi overrides if explicitly mentioned) ---------
    generic_emi = re.search(r'(under|below|less than|upto|<=)\s*([0-9]{3,8})\s*emi', cleaned_query)
    if generic_emi:
        filters['lowest_emi'] = {'lte': int(generic_emi.group(2))}
        cleaned_query = cleaned_query.replace(generic_emi.group(0), '').strip()

    # --------- 4. Generic price "under 35000" NOT followed by 'emi' ---------
    # If the query mentions "emi" anywhere, skip this block (handled above).
    # Otherwise, treat as price filter (MOP)
    if 'emi' not in cleaned_query and 'installment' not in cleaned_query and 'monthly' not in cleaned_query and 'per month' not in cleaned_query:
        general_patterns = [
            (r'(under|below|less than|upto|<=)\s*([0-9]{3,8})',
             lambda m: ('mop', {'lte': int(m.group(2))})),
            (r'(above|over|greater than|more than|>=)\s*([0-9]{3,8})',
             lambda m: ('mop', {'gte': int(m.group(2))})),
            (r'(from|between|range)?\s*([0-9]{3,8})\s*(to|and|-)\s*([0-9]{3,8})',
             lambda m: ('mop', {'gte': min(int(m.group(2)), int(m.group(4))), 'lte': max(int(m.group(2)), int(m.group(4)))}))
        ]
        for pat, rng_fn in general_patterns:
            for match in re.finditer(pat, cleaned_query):
                key, val = rng_fn(match)
                filters[key] = val
                cleaned_query = cleaned_query.replace(match.group(0), '').strip()

    # --------- 5. Fallback: number at end (if not already set), treat as price (mop lte) ---------
    if "mop" not in filters and "lowest_emi" not in filters:
        match = re.search(r'([0-9]{3,8})$', cleaned_query)
        if match:
            price = int(match.group(1))
            filters['mop'] = {'lte': price}
            cleaned_query = cleaned_query.replace(match.group(0), '').strip()

    # --- Clean up extra spaces ---
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    if not cleaned_query:
        cleaned_query = query

    return cleaned_query, (filters if filters else None)




#def expand_search_terms(user_query):
 #   user_query_n = normalize(user_query)
  #  user_query_ns = user_query_n.replace(" ", "")
   # variants = set([user_query_n, user_query_ns])
   # for key, syns in BUSINESS_SYNONYMS.items():
   #     if user_query_n == key or user_query_ns == key or user_query_n in syns or user_query_ns in syns:
   #         variants.add(key)
   #         variants.update(syns)
   # mapped = CATEGORY_CANONICAL.get(user_query_n) or CATEGORY_CANONICAL.get(user_query_ns)
   # if mapped: variants.add(mapped)
   # variants.update(user_query_n.split())
   # return sorted({t.strip() for t in variants if t.strip()})


def expand_search_terms(user_query):
    user_query_n = normalize(user_query)
    variants = set([user_query_n])
 
    # Always include canonical mapping (robust for synonyms and categories)
    canonical_category = CATEGORY_CANONICAL.get(user_query_n)
    if canonical_category:
        variants.add(canonical_category)
 
    # Add synonyms and all mapped forms
    for canon, syns in BUSINESS_SYNONYMS.items():
        if user_query_n == canon or user_query_n in syns:
            variants.add(canon)
            variants.update(syns)
 
    # Split by spaces and add as additional variants
    variants.update(user_query_n.split())
    return sorted({t.strip() for t in variants if t.strip()})

                                                                                                                                                                                                          




def update_image_url(product):
    for prod in product.get("products", []):
        image = prod.get("image", "")
        if image and not image.startswith("http"):
            if image.startswith("/"):
                prod["image"] = IMAGE_DOMAIN + image
            else:
                prod["image"] = IMAGE_DOMAIN + "/" + image
    return product




def resolve_category_for_exclusions(cat: str):
    if not cat:
        return "blank"
    cat_l = cat.lower()
    resolved = ALL_SYNONYM_TO_CANONICAL.get(cat_l, cat_l)
    return resolved if resolved in FILTER_ATTRIBUTE_EXCLUSIONS else "blank"

def clean_products_for_plp(response_dict):
    """Remove 'label' from attribute_swatch_color and the color/brand fields from each SKU in products list."""
    try:
        for product in response_dict["data"]["PostV1Productlist"]["data"]["products"]:
            for sku in product.get("products", []):
                # Remove fields outside attribute_swatch_color
                for key in ["brand_id", "color_hex_code", "color_label"]:
                    if key in sku:
                        del sku[key]
                # Remove 'label' inside attribute_swatch_color (dict or list of dicts)
                if "attribute_swatch_color" in sku:
                    color = sku["attribute_swatch_color"]
                    if isinstance(color, dict):
                        if "label" in color:
                            del color["label"]
                    elif isinstance(color, list):
                        for c in color:
                            if isinstance(c, dict) and "label" in c:
                                del c["label"]
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    return response_dict

def get_intent_attributes(query):
    query = query.lower()
    colors = ["black", "white", "silver", "blue", "red", "grey", "gold", "green", "yellow", "pink", "purple"]
    storages = ["256 gb", "512 gb", "128 gb", "64 gb", "32 gb", "16 gb", "8 gb", "4 gb", "1 tb", "2 tb"]
    sizes = ["43 inch", "50 inch", "55 inch", "65 inch", "32 inch", "40 inch", "1.5 ton", "2 ton", "6 kg", "7 kg", "8 kg", "9 kg", "10 kg"]
    models = ["pro", "max", "plus", "mini", "air", "smart", "ultra"]
    # Add any other attributes relevant to your business

    intent = {
        "colors": [c for c in colors if c in query],
        "storages": [s for s in storages if s in query],
        "sizes": [s for s in sizes if s in query],
        "models": [m for m in models if m in query]
    }
    return intent

def score_product(prod, intent):
    score = 0
    color = (prod.get("color") or prod.get("colour") or prod.get("attribute_color") or "").lower()
    storage = (prod.get("storage") or prod.get("attribute_storage") or prod.get("attribute_internal_storage") or "").lower()
    size = (prod.get("size") or prod.get("screen_size") or prod.get("attribute_screen_size_in_inches") or "").lower()
    model = (prod.get("model") or prod.get("variant") or prod.get("attribute_variant") or "").lower()

    # Robust: Sometimes product info is nested under "products" key (for multi-SKU)
    if "products" in prod and isinstance(prod["products"], list) and prod["products"]:
        main_sku = prod["products"][0]
        color = (main_sku.get("color") or main_sku.get("colour") or main_sku.get("attribute_color") or color).lower()
        storage = (main_sku.get("storage") or main_sku.get("attribute_storage") or main_sku.get("attribute_internal_storage") or storage).lower()
        size = (main_sku.get("size") or main_sku.get("screen_size") or main_sku.get("attribute_screen_size_in_inches") or size).lower()
        model = (main_sku.get("model") or main_sku.get("variant") or main_sku.get("attribute_variant") or model).lower()
    # Score based on match
    if any(c in color for c in intent["colors"]): score += 3
    if any(s in storage for s in intent["storages"]): score += 2
    if any(sz in size for sz in intent["sizes"]): score += 2
    if any(m in model for m in intent["models"]): score += 1
    return -score  # Negative so best match sorts first

def sort_products_by_intent(products, query):
    intent = get_intent_attributes(query)
    return sorted(products, key=lambda p: score_product(p, intent))




def process_response(data, total, city_id=None,emi_range=None):
    """
    Processes raw Elasticsearch hits into the correct PLP response structure,
    with robust city-wise filtering. Only products available in the requested city_id
    or in the default city_id=0 are included. Others are skipped.
    """
    final_response = {
        "data": {
            "PostV1Productlist": {
                "status": True,
                "message": "Success",
                "data": {
                    "products": [],
                    "totalrecords": total,
                    "suggested_search_keyword": "*",
                    "filters": [],
                }
            }
        }
    }
    emi_range = []
    final_filters = {}
    canonical_cat = "blank"
    for entry in data:
        cat = entry["_source"].get("actual_category", "")
        if cat:
            canonical_cat = resolve_category_for_exclusions(cat)
            break
    excluded_filters = FILTER_ATTRIBUTE_EXCLUSIONS.get(canonical_cat, set())

    
    seen_skus = set()
    for entry in data:
        entry = entry["_source"]
        entry = update_image_url(entry)
        
        

        # --------------- CITY LEVEL LOGIC (final robust block) ---------------
        input_city_id = str(city_id) if city_id is not None else "0"
        city_id_to_check = f"citi_id_{input_city_id}"

        cityids = entry.get("cityid", [])
        if isinstance(cityids, str):
            cityids = [cityids]

        # Only include products available in the requested city_id, or city_id_0.
        if city_id_to_check in cityids:
            selected_city = city_id_to_check
        elif "citi_id_0" in cityids:
            selected_city = "citi_id_0"
        else:
            continue  
        
        sku_id = entry.get("modelid","999999")
        if sku_id in seen_skus:
            continue
        seen_skus.add(sku_id)

        emi_range.append(entry.get("lowest_emi", 0))
        temp_dict = {
            "model_id": entry.get("modelid","999999"),
            "model_launch_date": entry.get("model_launch_date","1970-01-01"),
            "mkp_active_flag": entry.get("mkp_active_flag",0),
            "avg_rating": entry.get("avg_rating",0),
            "rating_count": entry.get("rating_count",0),
            "asset_category_id": entry.get("asset_category_id",0),
            "asset_category_name": entry.get("asset_category_name","UNKNOWN"),
            "manufacturer_id": entry.get("manufacturer_id",999),
            "manufacturer_desc": entry.get("manufacturer_desc","UNKNOWN"),
            "category_type": entry.get("category_type","UNKNOWN"),
            "mop": entry.get("mop",0),
            "property": [],
            "products": []
        }

        # Only set property for matched city
        property_dict = {
            "cityid": [selected_city],
            "transaction_count": entry.get("transaction_count",0),
            "lowest_emi": entry.get("lowest_emi",0),
            "mop": entry.get("mop", 0),
            "offer_price": entry.get("offer_price",0),
            "score": entry.get("score",0),
            "ty_page_count": entry.get("ty_page_count",0),
            "one_emi_off": entry.get("one_emi_off",0),
            "pdp_view_count": entry.get("pdp_view_count",0),
            "off_percentage": entry.get("off_percentage",0),
            "zero_dp_flag": entry.get("zero_dp_flag",0),
            "new_launch_flag": entry.get("new_launch_flag",0),
            "most_viewed_flag": entry.get("most_viewed_flag",0),
            "top_seller_flag": entry.get("top_seller_flag",0),
            "highest_tenure": entry.get("highest_tenure",0),
            "model_city_flag": entry.get("model_city_flag",0),
            "phone_setup": entry.get("phone_setup",0),
            "exchange_flag": entry.get("exchange_flag",0),
            "installation_flag": entry.get("installation_flag",0),
        }
        temp_dict["property"].append(property_dict)
        
        for sku_item in entry.get("products", []):
            temp_dict["products"].append(sku_item)

        final_response["data"]["PostV1Productlist"]["data"]["products"].append(temp_dict)
            
        attribute_name = [field.replace("_value","") for field in entry.keys() if (
            field.startswith("attribute_") and field.endswith("_value"))]
        for filter_entry in attribute_name:
            if filter_entry in excluded_filters:
                continue
            value = entry.get(f"{filter_entry}_value", "")
            fid = entry.get(filter_entry, "")
            if not value or value in ["", "UNKNOWN", None]: continue
            if filter_entry not in final_filters:
                final_filters[filter_entry] = set()
        # Special case for color swatch
            if "color_swatch" in filter_entry.lower() or "swatch_color" in filter_entry.lower():
                color_entries = set()
                for sku in entry.get("products", []):
                    color_info = sku.get("attribute_swatch_color", {})
                    color_hex = color_info.get("value") or ""
                    color_name = color_info.get("name") or ""
                    color_id = color_info.get("id") or fid or ""
                    if color_hex and color_name and color_id:
                        color_entries.add((f"{color_hex}_{color_name}", color_id))
                if not color_entries and value and fid:
                    color_entries.add((value, fid))
                for name_out, id_out in color_entries:
                    final_filters[filter_entry].add((name_out, id_out))
            else:
                final_filters[filter_entry].add((value, fid))


    # --- Compile final filters in required format ---
    for key, items in final_filters.items():
        unique_items = []
        seen_ids = set()
        for val, fid in items:
            if fid not in seen_ids:
                unique_items.append({"name": val, "id": fid})
                seen_ids.add(fid)
        final_filters[key] = unique_items
    final_filters = {k: v for k,v in final_filters.items() if len(v) > 0}
    
    # if "attribute_color_swatch"  in final_filters:
    #     final_filters["attribute_color_swatch_display_Color"] = final_filters.pop("attribute_color_swatch")
        
    if len(final_filters)>0:
        temp_final_filter = {}
        temp_final_filter["attributes"] = final_filters
        final_response["data"]["PostV1Productlist"]["data"]["filters"].append(temp_final_filter)

    # --- EMI filter (if present) ---
    if len(emi_range)>0 and len(final_response["data"]["PostV1Productlist"]["data"]["filters"]) > 0:
        min_emi = min(emi_range)
        max_emi = max(emi_range)
        final_response["data"]["PostV1Productlist"]["data"]["filters"][0]["emi"] = {
            "max": max_emi,
            "min": min_emi
        }
    return final_response


def is_mobile_query(query_n):
    all_mobiles_syns = set(BUSINESS_SYNONYMS.get("mobile phones", [])) | set([
        "mobile phones", "mobiles", "mobiles", "smartphone", "phone", "phones", "mobilephone", "cellphone", "cell phone"
    ])
    qn = query_n
    qn2 = query_n.replace(" ", "")
    return qn in all_mobiles_syns or qn2 in all_mobiles_syns

def is_apple_query(query_n):
    all_apple_syns = set(BUSINESS_SYNONYMS.get("apple", [])) | set([
        "apple", "iphone", "iphones", "apple phone", "apple phones", "apple mobile", "apple mobiles", "apple smartphone"
    ])
    qn = query_n
    qn2 = query_n.replace(" ", "")
    return qn in all_apple_syns or qn2 in all_apple_syns

def is_refurbished(prod):
    for field in [
        prod.get("actual_category", ""),
        prod.get("category_type", ""),
        prod.get("model_id", ""),
        prod.get("model_launch_date", ""),
        prod.get("manufacturer_desc", ""),
        prod.get("asset_category_name", ""),
        prod.get("product_name", ""),
    ]:
        if field and "refurbished" in field.lower():
            return True
    for sku in prod.get("products", []):
        if "refurbished" in (sku.get("name", "") + sku.get("sku", "")).lower():
            return True
    return False


def build_advanced_search_query(
    user_query,
    filters=None,
    city_id=None,
    mapped_category=None,
    price_filter=None,
    products_is_nested=True,
    emi_range=None,  # <-- ADD THIS
):

    terms = expand_search_terms(user_query)
    should_clauses = []
    must_clauses = []
    must_not_clauses = []
    category_boosts = []
    filter_clauses = []
    
    for term in terms:
        should_clauses += [
            {"match": {"product_name": {"query": term, "boost": 5}}},
            {"match": {"search_field": {"query": term, "boost": 5}}},
            {"match": {"actual_category": {"query": term, "boost": 4}}},
            {"match": {"top_level_category_name": {"query": term, "boost": 2}}},
            {"match": {"product_keywords": {"query": term, "boost": 2}}},
            {"match_phrase_prefix": {"search_field": {"query": term, "boost": 3}}},
            {"match": {"search_field": {"query": term, "fuzziness": "AUTO", "boost": 2}}},
            {"match": {"products.sku": {"query": term, "boost": 2}}},
            {"match": {"products.name": {"query": term, "boost": 2}}},
        ]

    # ------------- CATEGORY FALLBACK -------------
    if mapped_category:
        should_clauses += [
            {"term": {"actual_category": mapped_category}},
            {"match": {"actual_category": {"query": mapped_category, "boost": 4}}},
        ]

    parsed = decompose_query_for_boosting(user_query)
    # parsed = patch_categories_for_mobile(parsed)
    brandlock_must, brandlock_must_not = patch_brand_token_must(parsed, user_query)
    must_clauses.extend(brandlock_must)
    must_not_clauses.extend(brandlock_must_not)
    
    brands = parsed["brands"]
    categories = parsed["categories"]
    attributes = parsed["attributes"]
    
        
    if brands:
        for brand in brands:
            # Strong boost for direct brand field match
            should_clauses.append({
                "term": {
                    "manufacturer_desc": {
                        "value": brand,
                        "boost": 200
                    }
                }
            })
            # ALSO boost if brand appears in product_name (use match_phrase for precision)
            should_clauses.append({
                "match_phrase": {
                    "product_name": {
                        "query": brand,
                        "boost": 150
                    }
                }
            })
            # ALSO boost if brand appears in search_field (if you use one)
            should_clauses.append({
                "match_phrase": {
                    "search_field": {
                        "query": brand,
                        "boost": 120
                    }
                }
            })
            # correct speling mistake correct 
            should_clauses.append({
                "match": {
                    "product_name": {
                        "query": brand,
                        "fuzziness": "AUTO",
                        "boost": 80
                    }
                }
            })

            
    if categories:
        for cat in categories:    
            should_clauses.append({
                "term": {
                    "actual_category": {
                        "value": cat,
                        "boost": 100
                    }
                }
            })
            # Boost if category term is in product_name or search_field
            should_clauses.append({
                "match_phrase": {
                    "product_name": {
                        "query": cat,
                        "boost": 70
                    }
                }
            })
            should_clauses.append({
                "match_phrase": {
                    "search_field": {
                        "query": cat,
                        "boost": 50
                    }
                }
            })

    
    if attributes.get("color"):
        for color in attributes["color"]:
            # Boost on direct color attribute
            should_clauses.append({
                "term": {
                    "attribute_color": {
                        "value": color,
                        "boost": 80
                    }
                }
            })
            # Boost in product_name/search_field
            should_clauses.append({
                "match_phrase": {
                    "product_name": {
                        "query": color,
                        "boost": 60
                    }
                }
            })
            should_clauses.append({
                "match_phrase": {
                    "search_field": {
                        "query": color,
                        "boost": 50
                    }
                }
            })

    if attributes.get("storage"):
        for storage in attributes["storage"]:
            should_clauses.append({
                "term": {
                    "attribute_internal_storage": {
                        "value": storage,
                        "boost": 80
                    }
                }
            })
            should_clauses.append({
                "match_phrase": {
                    "product_name": {
                        "query": storage,
                        "boost": 60
                    }
                }
            })
            should_clauses.append({
                "match_phrase": {
                    "search_field": {
                        "query": storage,
                        "boost": 50
                    }
                }
            })

    if attributes.get("ram"):
        for ram in attributes["ram"]:
            should_clauses.append({
                "term": {
                    "attribute_ram": {
                        "value": ram,
                        "boost": 80
                    }
                }
            })
            should_clauses.append({
                "match_phrase": {
                    "product_name": {
                        "query": ram,
                        "boost": 60
                    }
                }
            })
            should_clauses.append({
                "match_phrase": {
                    "search_field": {
                        "query": ram,
                        "boost": 50
                    }
                }
            })



    # ---------- CATEGORY LOCKING ----------
    CATEGORY_LOCK_SYNS = {
    "car": set([
        "car", "cars", "new car", "new cars","for wheeler", "four wheeler", "four-wheeler", "4 wheeler",
        "skoda", "volkwagen", "hyundai", "bmw", "marcedes", "4wheeler", "sedan", "hatchback", "suv"
    ] + [normalize(x) for x in BUSINESS_SYNONYMS.get("new cars", [])]),

    "smartphone": set([
        "smartphone", "mobile", "mobiles", "phone", "phones", "cellphone", "cell phone"
    ] + [normalize(x) for x in BUSINESS_SYNONYMS.get("mobiles", [])]),

    "television": set([
        "television", "tv", "tvs", "led tv", "smart tv"
    ] + [normalize(x) for x in BUSINESS_SYNONYMS.get("tv and home entertainment", [])]),

    "laptop": set([
        "laptop", "laptops", "notebook", "ultrabook", "chromebook", "macbook"
    ] + [normalize(x) for x in BUSINESS_SYNONYMS.get("laptops", [])]),

    "tractor": set([
        "tractor", "tractors", "farm tractor", "agriculture tractor"
    ] + [normalize(x) for x in BUSINESS_SYNONYMS.get("tractor", [])]),

    "refrigerator": set([
        "refrigerator", "fridge", "freezer"
    ] + [normalize(x) for x in BUSINESS_SYNONYMS.get("refrigerators", [])]),

    "ac": set([
        "ac", "air conditioner", "airconditioner", "split ac", "window ac"
    ] + [normalize(x) for x in BUSINESS_SYNONYMS.get("ac", [])]),

    "washing machine": set([
        "washing machine", "washing machines", "Washing machine", "wahing", "kapde machine",
        "washer", "laundry machine", "clothes washer", "front load washer", "top load washer", "washingmachine"
    ] + [normalize(x) for x in BUSINESS_SYNONYMS.get("washing machines", [])])
    }
    category_locked = False
    lock_category = None
    qn = normalize(user_query)
    q_tokens = set(qn.split())
    
    # --- HARD CATEGORY LOCK FOR INVERTER/BACKUP QUERIES ---
    MOBILE_BLOCK_TERMS = [
        "blynk", "refurbished", "reled", "LED Smart Android","inverter split AC", "iphone xs", "iphone xs 64", "xs", "iphone xs max"
    ]
    if is_mobile_query(user_query):
        for block_term in MOBILE_BLOCK_TERMS:
            must_not_clauses.append({"match_phrase": {"product_name": block_term}})
            must_not_clauses.append({"match_phrase": {"manufacturer_desc": block_term}})
            must_not_clauses.append({"match_phrase": {"asset_category_name": block_term}})

    # 1. Define all user queries that should be "inverter" only:
    INVERTER_SYNONYMS = {
        "inverter", "inverters", "backup", "battery backup", "power backup",
        "inverter battery", "home inverter", "power inverter"
    }
    user_query_norm = normalize(user_query)
    if user_query_norm in INVERTER_SYNONYMS:
        # Remove other category locks, enforce inverter
        must_clauses = [cl for cl in must_clauses if cl.get("term", {}).get("actual_category") == "inverter"]
        must_clauses.append({"term": {"actual_category": "inverter"}})
        # Strongly exclude speakers, plates, power flash, emergency light, etc.
        must_not_clauses.extend([
            {"term": {"actual_category": "ac"}},
            {"term": {"actual_category": "air conditioner"}},
            {"match": {"product_name": "ac"}},
            {"match": {"product_name": "split ac"}},
            {"match": {"product_name": "speaker"}},
            {"match": {"product_name": "speakers"}},
            {"match": {"product_name": "built in battery"}},
            {"match": {"product_name": "battery plate"}},
            {"match": {"product_name": "battery plates"}},
            {"match": {"product_name": "power flash"}},
            {"match": {"product_name": "emergency light"}},
            {"match": {"product_name": "emergency bulb"}},
            {"match": {"product_name": "power bulb"}},
            {"match": {"product_name": "flashlight"}},
            {"match": {"product_name": "torch"}}
        ])

    
    

    for cat, syn_set in CATEGORY_LOCK_SYNS.items():
        if qn in syn_set or q_tokens & syn_set:
            must_clauses.append({"terms": {"actual_category": list(syn_set)}})
            category_locked = True
            lock_category = cat
            break
        
    CAR_SYNS = CATEGORY_LOCK_SYNS.get("car", set())
    if (qn in CAR_SYNS or q_tokens & CAR_SYNS) and not category_locked:
        must_clauses.append({"term": {"actual_category": "car"}})
        category_locked = True
        lock_category = "car"
        # Exclude accessories, tyres, coolers, etc.
        must_not_clauses.extend([
            {"term": {"actual_category": "tyre"}},
            {"term": {"actual_category": "car accessory"}},
            {"term": {"actual_category": "car cooler"}},
            {"match": {"product_name": "tyre"}},
            {"match": {"product_name": "accessory"}},
            {"match": {"product_name": "cooler"}},
        ])
        
        # PATCH: STRICT CATEGORY LOCKING for "washing machine" (covers variants)
    WASHING_MACHINE_CATEGORIES = [
    "washing machine", "Washing Machine", "washing machines", "WASHING_MACHINE", "WASHING_MACHINES"
    ]
    if not category_locked:
        if (
            qn in CATEGORY_LOCK_SYNS["washing machine"]
            or any(word in CATEGORY_LOCK_SYNS["washing machine"] for word in q_tokens)
            or "washing machine" in user_query.lower()
            or "washing machines" in user_query.lower()
        ):
            
            must_clauses.append({"terms": {"actual_category": WASHING_MACHINE_CATEGORIES}})
            category_locked = True
            lock_category = "washing machine"
            
            # --- Exclude laptops, washbasins and noisy categories ---
            must_not_clauses.extend([
                {"term": {"actual_category": "Laptop"}},
                {"term": {"actual_category": "Washbasin"}},
                {"match": {"product_name": "Washbasin"}},
                # Optionally exclude furniture if you notice more pollution
                # {"term": {"actual_category": "furniture"}},
            ])




        
    # if not category_locked:
    #     # Extra strictness for "washing machine" and its synonyms
    #     if qn in CATEGORY_LOCK_SYNS["washing machine"] or any(word in CATEGORY_LOCK_SYNS["washing machine"] for word in q_tokens):
    #         must_clauses.append({"term": {"actual_category": "washing machine"}})
    #         category_locked = True
    #         lock_category = "washing machine"
            
    if not category_locked:
        
        if qn in CATEGORY_LOCK_SYNS["tractor"] or any(word in CATEGORY_LOCK_SYNS["tractor"] for word in q_tokens):
            must_clauses.append({"term": {"actual_category": "tractor"}})
            category_locked = True
            lock_category = "tractor"
            
    if not category_locked:
        
        if qn in CATEGORY_LOCK_SYNS["smartphone"] or any(word in CATEGORY_LOCK_SYNS["smartphone"] for word in q_tokens):
            must_clauses.append({"term": {"actual_category": "smartphone"}})
            category_locked = True
            lock_category = "smartphone"
            
    q_scoot = {"scooter", "scooty", "activa", "jupiter", "vespa", "burgman", "maestro", "grazia", "access", "ntorq"}
    if any(word in q_tokens for word in q_scoot):
        must_clauses[:] = [cl for cl in must_clauses if cl.get("term", {}).get("actual_category") == "two-wheeler"]
        if not any(cl.get("term", {}).get("actual_category") == "two-wheeler" for cl in must_clauses):
            must_clauses.append({"term": {"actual_category": "two-wheeler"}})
        must_not_clauses.extend([
            {"term": {"actual_category": "air cooler"}},
            {"term": {"actual_category": "cooler"}},
            {"term": {"actual_category": "water heater"}},
            {"match_phrase": {"product_name": "cooler"}},
            {"match_phrase": {"product_name": "air cooler"}},
            {"match_phrase": {"product_name": "water heater"}},
            {"match_phrase": {"manufacturer_desc": "HERO"}},
            {"match_phrase": {"product_name": "Bajaj Pulsar"}},
            {"match_phrase": {"product_name": "TVS Radeon"}},
            {"match_phrase": {"product_name": "Hero Splendor Plus"}}
        ])

    # ---------- SMARTPHONE-FIRST BRAND BOOST ----------
    
    user_query_n = normalize(user_query)

    def is_brand_smartphone_query(user_query_n, brands):
        tokens = user_query_n.split()
        for brand in brands:
            if (
                brand in user_query_n and (
                    len(tokens) == 1 or
                    (len(tokens) == 2 and tokens[0] == brand and tokens[1] in {"mobile", "phone", "smartphone"}) or
                    (len(tokens) == 2 and tokens[1] == brand and tokens[0] in {"mobile", "phone", "smartphone"})
                )
            ):
                return brand
            if brand == "samsung" and ("galaxy" in tokens or "galaxy" in user_query_n):
                return brand
        return None

    brand_matched = is_brand_smartphone_query(user_query_n, MOBILE_BRANDS)
    if brand_matched:
        should_clauses.append({
            "bool": {
                "must": [
                    {"term": {"actual_category": {"value": "smartphone"}}},
                    {"term": {"manufacturer_desc": {"value": brand_matched, "boost": 100}}}
                ],
                "boost": 200
            }
        })
        should_clauses.append({
            "term": {"actual_category": {"value": "smartphone", "boost": 120}}
        })
        should_clauses.append({
            "term": {"manufacturer_desc": {"value": brand_matched, "boost": 120}}
        })

    # ---------- USUAL BOOSTING LOGIC ----------
    for cat in parsed["categories"]:
        should_clauses.append({
            "term": {"actual_category": {"value": cat, "boost": 15}}
        })
        should_clauses.append({
            "term": {"category_type": {"value": cat, "boost": 6}}
        })
        should_clauses.append({
            "term": {"asset_category_name": {"value": cat, "boost": 6}}
        })

    for brand in parsed["brands"]:
        should_clauses.append({
            "term": {"manufacturer_desc": {"value": brand, "boost": 10}}
        })

    for attr, vals in parsed["attributes"].items():
        for val in vals:
            should_clauses.append({
                "match": {"product_name": {"query": val, "boost": 6}}
            })
            should_clauses.append({
                "match": {"search_field": {"query": val, "boost": 5}}
            })

    if len(parsed["tokens"]) > 2:
        should_clauses.append({
            "match": {
                "product_name": {
                    "query": " ".join(parsed["tokens"]),
                    "operator": "and",
                    "boost": 10
                }
            }
        })

    # ------------- SPECIAL CASE HANDLING (existing logic) -------------
    all_mobiles_syns = set(BUSINESS_SYNONYMS.get("mobiles", [])) | set([
        "mobile phones", "mobiles", "smartphone", "phone", "phones", "mobilephone", "cellphone", "cell phone"
    ])
    user_query_n = normalize(user_query)
    user_query_ns = user_query_n.replace(" ", "")

    # --------- SPECIAL CASE HANDLING ---------
    
    
    
    # ----- PATCHED: Robust scooter/twowheeler matching -----
    SCOOTER_SYNONYMS_SET = {
        "scooter", "scooters", "scooty", "activa", "jupiter", "vespa", "burgman", "maestro", "grazia", "access", "ntorq"
    }
    if any(word in user_query_n.split() for word in SCOOTER_SYNONYMS_SET):
        must_clauses.append({"term": {"actual_category": "two-wheeler"}})
        # Block air coolers/cooler/fuzzy matches:
        must_not_clauses.extend([
            {"term": {"actual_category": "air cooler"}},
            {"term": {"actual_category": "cooler"}},
            {"term": {"actual_category": "water heater"}},
            {"match_phrase": {"product_name": "cooler"}},
            {"match_phrase": {"product_name": "air cooler"}},
            {"match_phrase": {"product_name": "water heater"}},
            {"match_phrase": {"manufacturer_desc": "HERO"}},
            {"match_phrase": {"product_name": "Hero Splendor Plus"}},
            {"match_phrase": {"product_name": "Honda SP"}},
            {"match_phrase": {"product_name": "TVS"}},
            
            
        ])
        should_clauses.extend([
            {"match": {"product_name": {"query": "activa", "boost": 20}}},
            {"match": {"product_name": {"query": "electric scooter", "boost": 12}}},
            {"match": {"manufacturer_desc": {"query": "tvs jupiter", "boost": 12}}},
            {"match": {"manufacturer_desc": {"query": "honda activa", "boost": 12}}},
            {"match": {"product_name": {"query": "scooty", "boost": 11}}},
            {"match": {"product_name": {"query": "vespa", "boost": 10}}},
            {"match": {"product_name": {"query": "burgman", "boost": 10}}},
            {"match": {"product_name": {"query": "maestro", "boost": 10}}},
            {"match": {"product_name": {"query": "ntorq", "boost": 10}}},
            {"match": {"product_name": {"query": "grazia", "boost": 10}}},
            {"match": {"product_name": {"query": "access", "boost": 10}}},
        ])

    if user_query_n in {"scooty", "scoty", "activa"}:
        must_clauses.append({"term": {"actual_category": "two-wheeler"}})
        must_not_clauses.extend([
            {"match_phrase": {"product_name": "splendor"}},
            {"match_phrase": {"product_name": "hero splendor"}},
            {"match_phrase": {"product_name": "bike"}},
            {"term": {"actual_category": "air cooler"}},
            {"term": {"actual_category": "water heater"}},
        ])
        should_clauses.extend([
            {"match": {"product_name": {"query": "activa", "boost": 12}}},
            {"match": {"product_name": {"query": "electric scooter", "boost": 10}}},
            {"match": {"manufacturer_desc": {"query": "tvs jupiter", "boost": 10}}},
            {"match": {"manufacturer_desc": {"query": "honda activa", "boost": 10}}},
        ])
        
    elif user_query_n in {"hero bike", "honda bike","hero honda bike"}:
        must_clauses.append({"term": {"actual_category": "two-wheeler"}})
        must_not_clauses.extend([
            {"match_phrase": {"product_name": "scooter"}},
            {"match_phrase": {"product_name": "activa"}},
            {"match_phrase": {"product_name": "scooty"}},
            {"match_phrase": {"product_name": "pleasure plus"}},
            {"match": {"product_name": {"query": "royal enfield", "boost": 8}}},
        ])
        should_clauses.extend([
            {"match": {"product_name": {"query": "hero honda", "boost": 12}}},
            {"match": {"product_name": {"query": "hero splendor", "boost": 12}}},
            {"match": {"manufacturer_desc": {"query": "hero", "boost": 12}}},
            
        ])
        
    elif user_query_n in {"bike", "hero bike", "honda bike","hero honda bike"}:
        must_clauses.append({"term": {"actual_category": "two-wheeler"}})
        must_not_clauses.extend([
            {"match_phrase": {"product_name": "scooter"}},
            {"match_phrase": {"product_name": "activa"}},
            {"match_phrase": {"product_name": "scooty"}},
            {"match_phrase": {"product_name": "pleasure plus"}},
        ])
        should_clauses.extend([
            {"match": {"product_name": {"query": "hero honda", "boost": 12}}},
            {"match": {"product_name": {"query": "hero splendor", "boost": 12}}},
            {"match": {"product_name": {"query": "bike", "boost": 10}}},
            {"match": {"manufacturer_desc": {"query": "bajaj pulsar", "boost": 10}}},
            {"match": {"manufacturer_desc": {"query": "hero", "boost": 12}}},
            {"match": {"product_name": {"query": "splendor", "boost": 15}}},
            # 
        ])
    if user_query_n in {"air purifier", "airpurifier", "air-purifier"}:
        must_clauses.append({"term": {"actual_category": "air purifier"}})
        must_not_clauses.extend([
            {"match_phrase": {"product_name": "water purifier"}},
            {"match_phrase": {"actual_category": "water purifier"}},
        ])
        
    if user_query_n in {"mop", "Mop", "mop cleaner"}:
        must_clauses.append({"term": {"actual_category": "vacuum cleaner"}})
        must_not_clauses.extend([
            {"match_phrase": {"product_name": "burner"}},
            {"match_phrase": {"product_name": "royal"}},
            {"match_phrase": {"product_name": "Top"}},
            {"match_phrase": {"product_name": "Freezer"}},
            
        ])
        
    
        
    if user_query_n in {"goole", "pixel", "pixel phone", "google phone", "google pixel", "google pixle", "gogle pixal"}:
        must_clauses.append({"term": {"actual_category": "smartphone"}})
        must_clauses.append({"term": {"manufacturer_desc": "Google"}})
        should_clauses.extend([
            {"match": {"product_name": {"query": "google pixel", "boost": 12}}}
            # 
        ])
        must_not_clauses.extend([
            {"match_phrase": {"product_name": "samsung"}},
            {"match_phrase": {"product_name": "blynk"}}])
        
    import re
    nothing_brand_pattern = re.compile(r"\bnothing\b", re.IGNORECASE)
    phone_pattern = re.compile(r"\bphone\b", re.IGNORECASE)

    if (
        ("nothing" in user_query_n and "phone" in user_query_n)
        or re.search(r"\bnothing\s?\d*\b", user_query_n)
        or user_query_n in {"nothing", "nothing smartphone",}
    ):
        must_clauses.append({"term": {"actual_category": "smartphone"}})
        must_clauses.append({"term": {"manufacturer_desc": "Nothing"}})
        should_clauses.extend([
            {"match": {"product_name": {"query": "nothing phone", "boost": 30}}},
            {"match": {"product_name": {"query": "nothing", "boost": 12}}},
            {"match": {"search_field": {"query": "nothing phone", "boost": 10}}},
        ])
        must_not_clauses.extend([
            {"term": {"manufacturer_desc": b}} for b in BRANDS if b != "nothing"
        ])
        
    if (
        ("washing" in user_query_n and "machine" in user_query_n)
        or re.search(r"\bwashing\s?\d*\b", user_query_n) or 
        re.search(r"\bmachine\s?\d*\b", user_query_n)
        or user_query_n in {"washing", "washing machine", "washing machines"}
    ):
        must_clauses.append({"term": {"actual_category": "washing machine"}})
        #must_clauses.append({"term": {"manufacturer_desc": "Nothing"}})
        must_not_clauses.extend([
            {"match": {"product_name": {"query": "Washbasin", "boost": 30}}},
            {"match": {"product_name": {"query": "laptop", "boost": 12}}},
            {"match": {"actual_category": {"query": "laptop", "boost": 10}}},
        ])
        
    
        
   
        
    if (
        ("smartphone" in user_query_n and "phone" in user_query_n)
        or re.search(r"\bsmartphone\s?\d*\b", user_query_n) or 
        re.search(r"\bphone\s?\d*\b", user_query_n)
        or user_query_n in {"mobile", "smartphone", "phone", "mobiles", "phones"}
    ):
        must_clauses.append({"term": {"actual_category": "smartphone"}})
        #must_clauses.append({"term": {"manufacturer_desc": "Nothing"}})
        must_not_clauses.extend([
            {"match": {"product_name": {"query": "iPhone", "boost": 30}}},
            {"match": {"product_name": {"query": "Blynk", "boost": 30}}},
            {"match": {"product_name": {"query": "iphone", "boost": 12}}},
            {"match": {"manufacturer_desc": {"query": "Apple", "boost": 10}}},
            {"match": {"manufacturer_desc": {"query": "Blynk", "boost": 10}}},
        ])
        # must_not_clauses.extend([
        #     {"term": {"manufacturer_desc": b}} for b in BRANDS if b != "nothing"
        # ])
        
    if (
        ("bike" in user_query_n and "motorcycle" in user_query_n)
        or re.search(r"\bbike\s?\d*\b", user_query_n) or 
        re.search(r"\bmotorcycle\s?\d*\b", user_query_n) or re.search(r"\bmotor\s?\d*\b", user_query_n)
        or user_query_n in {"motorcycle", "bike", "two wheeler", "motor cycle"}
    ):
        must_clauses.append({"term": {"actual_category": "two-wheeler"}})
        #must_clauses.append({"term": {"manufacturer_desc": "Nothing"}})
        must_not_clauses.extend([
            {"match": {"product_name": {"query": "FitX bike", "boost": 30}}},
            {"match": {"product_name": {"query": "mountain bike", "boost": 30}}},
            {"match": {"product_name": {"query": "gym bike", "boost": 12}}},
            {"match": {"product_name": {"query": "activa", "boost": 12}}},
            {"match": {"product_name": {"query": "sports activa", "boost": 12}}},
            
        ])


    # ------------- MAIN CATEGORY/SYNONYM BLOCK -------------
    is_mobile = user_query_n in all_mobiles_syns or user_query_ns in all_mobiles_syns
    if is_mobile:
        brand_boosts = [
            ("Samsung", 120),
            ("Oppo", 100),
            ("Vivo", 90),
            ("Realme", 80),
            ("Redmi", 60),
            ("Nothing", 50),
        ]
        for brand, boost in brand_boosts:
            should_clauses.append({
                "term": {"manufacturer_desc": {"value": brand, "boost": boost}}
            })
        must_clauses.append({"term": {"actual_category": "smartphone"}})
    elif user_query_n in set(TWOWHEELER_AUTOSUGGEST + BUSINESS_SYNONYMS["two-wheeler"] + ["two wheeler", "two-wheeler", "twowheeler"]) or user_query_ns in set(TWOWHEELER_AUTOSUGGEST + BUSINESS_SYNONYMS["two-wheeler"]):
        category_boosts.append({"term": {"actual_category": {"value": "two-wheeler", "boost": 16}}})
        # category_boosts.append({"term": {"search_field": {"value": "hero honda", "boost": 12}}})
        # category_boosts.append({"term": {"product_name": {"value": "activa", "boost": 12}}})
    elif user_query_n in set(BUSINESS_SYNONYMS["new cars"] + ["four wheeler", "four-wheeler", "for wheeler","fourwheeler", "car", "cars"]) or user_query_ns in set(BUSINESS_SYNONYMS["new cars"]):
        category_boosts.append({"term": {"actual_category": {"value": "new cars", "boost": 10}}})
        should_clauses.extend([
            {"match": {"product_name": {"query": "car", "boost": 30}}},
            {"match": {"product_name": {"query": "skoda sedan", "boost": 12}}},
            {"match": {"product_name": {"query": "skoda sedan", "boost": 12}}},
            {"match": {"product_name": {"query": "volkwagen suv", "boost": 12}}},
            {"match": {"product_name": {"query": "maruti hatchback", "boost": 10}}},
            {"match": {"search_field": {"query": "car", "boost": 15}}},
        ])
        # should_clauses.extend([
        #     {"match": {"product_name": {"query": "nothing phone", "boost": 30}}},
        #     {"match": {"product_name": {"query": "nothing", "boost": 12}}},
        #     {"match": {"search_field": {"query": "nothing phone", "boost": 10}}},
        # ])
        
        
        # category_boosts.append({"term": {"product_name": {"value": "skoda", "boost": 10}}})
        # category_boosts.append({"term": {"product_name": {"value": "Mahindra", "boost": 10}}})
        # category_boosts.append({"term": {"product_name": {"value": "hatchback", "boost": 10}}})
        # category_boosts.append({"term": {"products.sku": {"value": "muv car", "boost": 7}}})
        # category_boosts.append({"term": {"products.sku": {"value": "suv", "boost": 7}}})
        # category_boosts.append({"term": {"products.sku": {"value": "sedan", "boost": 7}}})

    # ------------- TERM EXPANSION -------------
    for term in terms:
        should_clauses += [
            {"match": {"product_name": {"query": term, "boost": 5}}},
            {"match": {"search_field": {"query": term, "boost": 5}}},
            {"match": {"actual_category": {"query": term, "boost": 4}}},
            {"match": {"top_level_category_name": {"query": term, "boost": 2}}},
            {"match": {"product_keywords": {"query": term, "boost": 2}}},
            {"match_phrase_prefix": {"search_field": {"query": term, "boost": 3}}},
            {"match": {"search_field": {"query": term, "fuzziness": "AUTO", "boost": 2}}},
            {"match": {"products.sku": {"query": term, "boost": 2}}},
            {"match": {"products.name": {"query": term, "boost": 2}}},
        ]

    # ------------- CATEGORY FALLBACK -------------
    if mapped_category:
        should_clauses += [
            {"term": {"actual_category": mapped_category}},
            {"match": {"actual_category": {"query": mapped_category, "boost": 4}}},
        ]

    # ------------- CITY CLAUSE -------------
    if city_id:
        filter_clauses.append({
            "bool": {
                "should": [
                    {"term": {"cityid": city_id}},
                    {"term": {"cityid": "citi_id_0"}}
                ],
                "minimum_should_match": 1
            }
        })

    # ------------- FILTER CLAUSES (SAFE) -------------
# ------------- FILTER CLAUSES (FLAT - CORRECT) -------------
    if filters:
        for entry in filters:
            if entry == "emi":
                continue
            values = [v.strip() for v in str(filters[entry]).split(",") if v.strip()]
            if not values:
                continue
            filter_clauses.append({"terms": {entry: values}})


    # ------------- PRICE FILTER -------------
    if price_filter:
        for field, rng in price_filter.items():
            filter_clauses.append({"range": {field: rng}})
            
    if emi_range:
        filter_clauses.append({"range": {"lowest_emi": emi_range}})


    # ------------- FINAL QUERY -------------
    es_query = {
        "bool": {
            "filter": filter_clauses,
            "must": must_clauses,
            "must_not": must_not_clauses,
            "should": should_clauses + category_boosts,
            "minimum_should_match": 1
        }
    }

    # ------- Optional: Log for debug -------
    import json
    print("\n[DEBUG] FINAL ES QUERY BODY:\n", json.dumps(es_query, indent=2), "\n")

    return es_query




def resolve_query_and_filters(data, es=None, mapped_category_in=None):
    """
    Combines user query and chip (if present) for robust search and filter logic.
    Handles canonical category mapping, chip-to-attribute mapping, SKU/variant mapping, and edge/corner flexis.
    """
    query = data.get("query", "").strip()
    chip = data.get("chip", "").strip()
    base_category = data.get("base_category", "").strip()
    filters = data.get("filters", {}).copy()
    mapped_category = mapped_category_in  # Use if already set, else None

    # Normalize everything for robustness
    base_cat_norm = normalize(base_category)
    query_norm = normalize(query)
    chip_norm = normalize(chip)

    # --- 1. Canonical mapping using synonyms and canonical dict ---
    def resolve_canonical(q):
        qn = normalize(q)
        # Try BUSINESS_SYNONYMS first
        for canon, syns in BUSINESS_SYNONYMS.items():
            if qn == normalize(canon) or qn in [normalize(s) for s in syns]:
                return canon
        # Try canonical mapping direct
        for k in CATEGORY_CANONICAL.keys():
            if qn == normalize(k) or qn == normalize(CATEGORY_CANONICAL[k]):
                return k
        return None

    mapped_category = mapped_category_in
    if base_category:
        mapped_category = resolve_canonical(base_category)
    if not mapped_category and query:
        mapped_category = resolve_canonical(query)

    # --- 2. Chip-to-attribute mapping for all supported categories ---
    category_attribute_map = {
    # Mobiles & Smartphone
    "mobile phones": "attribute_internal_storage",
    "smartphone": "attribute_internal_storage",

    # Refrigerators
    "refrigerators": "attribute_capacity_litres",
    "refrigerator": "attribute_capacity_litres",

    # AC & Air Conditioner
    "ac": "attribute_capacity_in_tons",
    "air conditioner": "attribute_capacity_in_tons",

    # Two-Wheeler (UPDATED)
    "two-wheeler": "attribute_engine_capacity_new",
    "twowheeler": "attribute_engine_capacity_new",
    "bike": "attribute_engine_capacity_new",
    "scooter": "attribute_engine_capacity_new",

    # New Cars (UPDATED)
    "new cars": "attribute_engine_capacity_4w",
    "car": "attribute_engine_capacity_4w",
    "cars": "attribute_engine_capacity_4w",

    # Laptops
    "laptops": "attribute_storage_size",
    "laptop": "attribute_storage_size",
    # SSD capacity (add if you want to support chips for SSD specifically)
    "laptop_ssd": "attribute_ssd",

    # Washing Machines (UPDATED)
    "washing machines": "attribute_capacity_wm",
    "washing machine": "attribute_capacity_wm",

    # TV & Home Entertainment
    "tv and home entertainment": "attribute_screen_size_in_inches",
    "television": "attribute_screen_size_in_inches",

    # Audio & Video
    "audio & video": "attribute_speaker_weight",
    "audio and video": "attribute_speaker_weight",

    # Tractor
    "tractor": "attribute_engine_capacity",

    # Air Coolers (NEW)
    "air coolers": "attribute_capacity_air_cooler",
    "air cooler": "attribute_capacity_air_cooler",

    # Tablets (NEW)
    "tablets": "attribute_primary_camera_new",
    "tablet": "attribute_primary_camera_new",

    # Air Purifiers (NEW)
    "air purifiers": "attribute_suitable_for",
    "air purifier": "attribute_suitable_for",

    # Printers (NEW)
    "printers": "attribute_duplex_printing_new",
    "printer": "attribute_duplex_printing_new",

    # Desktop & Monitor (NEW)
    "desktop monitor": "attribute_screen_size_new",
    "monitor": "attribute_screen_size_new",

    # Furniture (left unchanged, no chip field in your data)
    "furniture": "attribute_type2",
}

    # Dynamic support for chip-to-attribute in all known chip-enabled categories
    if chip and mapped_category:
        attr_key = category_attribute_map.get(mapped_category)
        if not attr_key:
            # Fallback: check if mapped_category is in chips and try to infer
            if mapped_category in CATEGORY_HARDCODED_CHIPS:
                chips, _ = CATEGORY_HARDCODED_CHIPS[mapped_category]
                if chips:
                    attr_key = f"attribute_{normalize(mapped_category).replace(' ', '_')}"
        if attr_key:
            filters[attr_key] = chip
        else:
            # Optional: log missing mapping for dev review
            pass  # No-op for future proofing

    # --- 3. Chip as SKU/variant/color-level filter (optional advanced case) ---
    if chip and es is not None and mapped_category:
        sku_match_attr = ["color", "variant", "sku", "name"]
        # Heuristic: only attempt ES lookup for relevant chip size (avoid single-char chips, etc.)
        if len(chip_norm) > 1:
            try:
                query_body = {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"actual_category": mapped_category}},
                                {"multi_match": {"query": chip, "fields": [f"products.{field}" for field in sku_match_attr]}}
                            ]
                        }
                    },
                    "size": 1
                }
                sku_result = es.search(index=PRODUCT_INDEX_NAME, body=query_body)
                if sku_result.get("hits", {}).get("hits"):
                    filters["sku_chip"] = chip
            except Exception as ex:
                # Fail gracefully in edge/corner flexis (bad ES state, missing index, etc.)
                pass

    # --- 4. Combine base query and chip for "full intent" search (always combine if both are present) ---
    base_for_query = base_category if base_category else query
    if chip and base_for_query:
        # Combine for full semantic intent
        query_out = f"{base_for_query} {chip}".strip()
    else:
        query_out = base_for_query.strip()

    # --- 5. Absolute fallback: if nothing, return minimal fields (handles edge/corner flexis) ---
    if not query_out:
        query_out = chip or ""
    if not filters:
        filters = {}

    # --- 6. (Optional) Print debug ---
    # print("[DEBUG] query_out:", query_out, "| filters:", filters, "| mapped_category:", mapped_category)

    return query_out, filters, mapped_category


##############Muti words Query logic #########################


MOBILE_BRANDS = set([
    "samsung", "oppo", "vivo", "google pixel", "google pixel3", "realme", "redmi", "oneplus", "xiaomi", "apple", "nothing", "motorola", "iqoo", "infinix", "tecno"
])

MOBILE_ATTRS = set(["storage", "ram", "color", "battery", "screen", "camera"])
from rapidfuzz import process as fuzzproc, fuzz
def patch_categories_for_mobile(q_decomposed):
    # Only add category if not already present
    if not q_decomposed["categories"]:
        has_brand = any(b in MOBILE_BRANDS for b in q_decomposed["brands"])
        has_mobile_attr = any(attr in MOBILE_ATTRS for attr in q_decomposed["attributes"])
        if has_brand and has_mobile_attr:
            q_decomposed["categories"].append("smartphone")
    return q_decomposed

BRANDS = [
        "samsung", "apple", "oppo", "vivo", "redmi", "realme", "oneplus", "nokia", "xiaomi", "motorola", "lg",
        "panasonic", "sony", "lenovo", "hp", "dell", "acer", "asus", "mi", "iqoo", "poco", "infinix", "lava", "micromax"
    ]
COLOR_SYNONYMS = {
    "red": ["red", "redd", "red color", "red colur", "reddish", "laal", "lal"],
    "black": ["black", "blak", "black color", "black colur", "kala", "kala color", "kaala"],
    "blue": ["blue", "blu", "blue color", "blue colur", "neela", "nila"],
    "white": ["white", "whte", "white color", "white colur", "safed", "safed color"],
    "green": ["green", "gren", "green color", "hara", "hara color"],
    "gold": ["gold", "golden", "gold color", "sona", "sona color"],
    "grey": ["grey", "gray", "grey color", "gray color", "dhusar", "dhoosar"],
    "silver": ["silver", "slvr", "silver color", "chandi", "chandi color"],
    "pink": ["pink", "pnik", "pink color", "gulabi", "gulabi color"],
    "purple": ["purple", "parple", "purple color", "baingani"],
    "yellow": ["yellow", "yello", "yellow color", "peela", "peela color"],
    # add more as needed
}
COLOR_LOOKUP = {}
for canon, syns in COLOR_SYNONYMS.items():
    for s in syns:
        COLOR_LOOKUP[s] = canon
        
CATEGORIES = [
    "mobile phones", "mobiles", "smartphone", "phone", "laptop", "laptops", "notebook", "ultrabook",
    "television", "tv", "tvs", "refrigerator", "fridge", "ac", "air conditioner", "airconditioner", "split ac",
    "washing machine", "washing machines", "smartwatch", "smart watch", "camera", "tablet", "freezer"
]
# You should have a mapping of attribute regex patterns for each product vertical, here's a mini version:
ATTRIBUTE_PATTERNS = {
    
    "storage": r"(\d{2,4})\s?gb",
    "ram": r"(\d{1,2})\s?gb\s?ram",
    "screen": r"(\d{1,2}(\.\d)?(\s)?(inch|in|inches))",
    "battery": r"(\d{3,5})\s?mah",
    "camera": r"(\d{1,3})\s?mp",
    "engine_cc": r"(\d{3,5})\s?cc",
    "display": r"(hd|full hd|fhd|4k|amoled|lcd|oled|retina)",
    "variant": r"(pro|max|plus|mini|air|ultra|prime|sport|classic|trend)",
    "color": r"([a-z]+ color|[a-z]+ colou?r|[a-z]+)",

    # "storage": r"\b(\d{2,4}\s?gb)\b",
    # "ram": r"\b(\d{1,2}\s?gb\s?ram)\b",
    # "screen": r"\b(\d{1,2}(\.\d)?\s?(inch|in|inches))\b",
    # "battery": r"\b(\d{3,5}\s?mah)\b",
    # "camera": r"\b(\d{1,3}\s?mp)\b",
    # "color": r"\b(black|white|blue|red|green|yellow|gold|silver|grey|gray|pink|purple|beige|graphite)\b"

}
    
def decompose_query_for_boosting(query):
    norm_query = normalize(query)
    tokens = norm_query.split()
    brands_found = []
    categories_found = []
    attributes_found = {}

    # Used for multi-word/compound attributes
    query_words = norm_query

    # --- 1. Color Extraction (fuzzy & synonyms) ---
    detected_colors = set()
    for word in tokens:
        color_val = COLOR_LOOKUP.get(word)
        if not color_val:
            match, score, _ = fuzzproc.extractOne(word, COLOR_LOOKUP.keys(), scorer=fuzz.ratio, score_cutoff=78) or (None, 0, None)
            if match:
                color_val = COLOR_LOOKUP[match]
        if color_val:
            detected_colors.add(color_val)
    # Also scan multi-word for color phrases ("red color", "black colur", etc)
    for canon, syns in COLOR_SYNONYMS.items():
        for syn in syns:
            if syn in query_words:
                detected_colors.add(canon)
    if detected_colors:
        attributes_found["color"] = list(detected_colors)

    # --- 2. Storage, RAM, Battery, etc. ---
    for attr, pat in ATTRIBUTE_PATTERNS.items():
        if attr == "color": continue  # already handled
        matches = re.findall(pat, query_words)
        vals = []
        for m in matches:
            if isinstance(m, tuple):
                vals.append(m[0])
            else:
                vals.append(m)
        if vals:
            attributes_found[attr] = vals

    # --- 3. Display keywords (HD, 4K, AMOLED, etc) ---
    for disp_kw in ["hd", "full hd", "fhd", "4k", "amoled", "lcd", "oled", "retina"]:
        if disp_kw in query_words:
            attributes_found.setdefault("display", []).append(disp_kw)

    # --- 4. Brand Extraction (fuzzy) ---
    for word in tokens:
        brand, score, _ = fuzzproc.extractOne(word, BRANDS, scorer=fuzz.ratio, score_cutoff=80) or (None, 0, None)
        if brand and brand not in brands_found:
            brands_found.append(brand)
    # Also match multi-word brand tokens
    for brand in BRANDS:
        if brand in query_words and brand not in brands_found:
            brands_found.append(brand)

    # --- 5. Category Extraction (fuzzy) ---
    for word in tokens:
        cat, score, _ = fuzzproc.extractOne(word, CATEGORIES, scorer=fuzz.ratio, score_cutoff=80) or (None, 0, None)
        if cat and cat not in categories_found:
            categories_found.append(cat)
    for cat in CATEGORIES:
        if cat in query_words and cat not in categories_found:
            categories_found.append(cat)

    
    tokens_cleaned = [
        tok for tok in tokens
        if tok not in brands_found and tok not in categories_found and tok not in detected_colors
    ]
    
    nothing_found = not (brands_found or categories_found or attributes_found)
    if nothing_found:
        # All tokens become fallback keywords for the ES query
        fallback_keywords = tokens_cleaned if tokens_cleaned else tokens
        return {
            "brands": [],
            "categories": [],
            "attributes": {},
            "tokens": fallback_keywords  # these will go to ES query as boosted 'should' match
        }

    # If normal results found:
    return {
        "brands": list(set(brands_found)),
        "categories": list(set(categories_found)),
        "attributes": attributes_found,
        "tokens": tokens_cleaned
    }




def parse_query_to_filters(query):
    # This is a simplified version. For prod, use spaCy or custom NER+pattern rules.
    q = query.lower()
    filters = {}
    sort_by = None

    # Brands
    for brand in BRANDS:
        if brand in q:
            filters['manufacturer_desc'] = brand
            q = q.replace(brand, "")

    # Categories
    for cat in CATEGORIES:
        if cat in q:
            filters['actual_category'] = CATEGORY_CANONICAL.get(cat, cat)
            q = q.replace(cat, "")

    # RAM
    match = re.search(r'(\d{1,2})\s*gb\s*ram', q)
    if match:
        filters['attribute_ram'] = match.group(1) + ' GB'
        q = q.replace(match.group(0), "")

    # Storage
    match = re.search(r'(\d{2,4})\s*gb(?!\s*ram)', q)
    if match:
        filters['attribute_internal_storage'] = match.group(1) + ' GB'
        q = q.replace(match.group(0), "")

    # Battery
    match = re.search(r'(\d{3,5})\s*mah', q)
    if match:
        filters['attribute_battery_capacity_new'] = match.group(1) + ' mAh'
        q = q.replace(match.group(0), "")

    # Capacity (fridge/AC)
    match = re.search(r'(\d{2,4})\s*litre', q)
    if match:
        filters['attribute_capacity_litres'] = match.group(1)
        q = q.replace(match.group(0), "")

    match = re.search(r'(\d\.\d|\d+)\s*ton', q)
    if match:
        filters['attribute_capacity_in_tons'] = match.group(1)
        q = q.replace(match.group(0), "")

    # 5G
    if "5g" in q:
        filters['product_keywords'] = "5g"
        q = q.replace("5g", "")

    # "Best", "Top", "Trending" => sort by rating/popularity
    if any(w in q for w in ["best", "top", "trending"]):
        sort_by = "popularity"  # or "avg_rating", as per your schema

    # Numeric price/emi filter (already handled in your code)
    _, price_emi_filters = parse_price_from_query(query)
    if price_emi_filters:
        filters.update(price_emi_filters)

    return filters, sort_by



######################EXACT MATCH TOP RESULTS#############

def boost_exact_matches(products, user_query):
    """
    Sort products so that exact matches on product_name or other key fields come first.
    Boosts:
      - Exact (case-insensitive) product_name == user_query
      - Exact modelid/SKU match (optional)
    """
    n_query = normalize(user_query)
    def exact_score(prod):
        # Product Name match (string match, normalized)
        prod_name = (prod.get("product_name") or "").strip().lower()
        if normalize(prod_name) == n_query:
            return -100  # Highest priority

        # Main SKU name match (if available)
        for sku in prod.get("products", []):
            sku_name = (sku.get("name") or "").strip().lower()
            if normalize(sku_name) == n_query:
                return -90
        # Optionally, exact modelid match (if user typed a modelid)
        modelid = (prod.get("model_id") or prod.get("modelid") or "").strip()
        if modelid and modelid == user_query.strip():
            return -80
        return 0  # Default, not an exact match

    # Sort: first by exact match score, then by any prior intent-based sort
    return sorted(products, key=lambda p: (exact_score(p),), reverse=False)






###########------------------API LOG-------------###################

import functools

def log_api_call(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        try:
            req_data = None
            try:
                req_data = request.get_json(force=True)
            except Exception:
                req_data = request.get_data().decode("utf-8")
            response = fn(*args, **kwargs)
            status = response[1] if isinstance(response, tuple) else 200
            end = time.time()
            duration_ms = int((end - start) * 1000)
            # Log what matters: endpoint, status, duration, top-level params
            log_obj = {
                "endpoint": request.path,
                "method": request.method,
                "status": status,
                "duration_ms": duration_ms,
                "remote_addr": request.remote_addr,
                "query": req_data,
            }
            msg = "[KPI-API] " + json.dumps(log_obj)
            logging.info(msg)
            # Print for tmux/debug if needed:
            # print(msg)  # <--- Uncomment for tmux
            return response
        except Exception as ex:
            end = time.time()
            logging.error("[KPI-API-ERROR] %s | %s" % (request.path, str(ex)))
            # print("[KPI-API-ERROR]", request.path, str(ex)) # Uncomment for tmux
            raise
    return wrapper

#######----------------------------search end poin-----------#

def preprocess_query(query):
    q = query.lower()
    q = q.replace("kapde machine", "washing")
    q = q.replace("kapda machine", "washing")
    q = q.replace("washing machine", "washing")
    q = q.replace("washing machines", "washing")
    q = q.replace("scooter", "activa")
    q = q.replace("scooty", "activa")
    q = q.replace("treadmill", "tread")
    q=  q.replace("treadmills", "tread")
    q = q.replace("tredmill", "tread")
    q = q.replace("tredmills", "tread")
    q = q.replace("tread-mill", "tread")
    q = q.replace("tred-mill", "tread")
    q = q.replace("tredmill", "tread")
    q = q.replace("tread mill", "tread")
    q = q.replace("flour mill", "flour")
    q = q.replace("flourmill", "flour")
    q = q.replace("flour-mill", "flour")
    q = q.replace("flour mills", "flour")
    q = q.replace("flor mills", "flour")
    q = q.replace("flor mill", "flour")
    q = q.replace("one plus", "oneplus")
    q = q.replace("microwave", "oven")
    q = q.replace("oppo mobile", "oppo")
    q = q.replace("oppo smartphone", "oppo")
    q = q.replace("vivo smartphone", "vivo")
    q = q.replace("nothing phone", "nothing")
    q = q.replace("nothing phones", "nothing")
    q = q.replace("nothing smartphones", "nothing")
    q = q.replace("nothing smartphone", "nothing")
    # q = q.replace("google", "googgle phone")
    # q = q.replace("google phone", "googgle phone")
    # q = q.replace("google mobile", "googgle phone")
    # q = q.replace("google pixel", "googgle phone")
    # q = q.replace("google picel", "googgle phone")
    # q = q.replace("googe picel", "googgle phone")
    # q = q.replace("gogle picel", "googgle phone")
    # q = q.replace("goggle picel", "googgle phone")
    # q = q.replace("gogle picel", "googgle phone")
    # q = q.replace("googe pixel", "googgle phone")
    #q = q.replace("pixel", "google phone")
    #q = q.replace("one", "oneplus")
    
 
    return q



# --- Stopword list (expand as needed) ---
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being below between both
but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't
has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm
i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other
ought our ours ourselves out over own same shan shan't she she'd she'll she's should shouldn't so some such than that that's
the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under
until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom
why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

def remove_stopwords(text):
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return ' '.join(tokens)

def stem_rule(token):
    # Example: very lightweight stemming (expand as needed)
    token = token.replace("colours", "color")
    if token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token

def rank_exact_query_first(products, user_query):
    norm_query = normalize(user_query)
    exact = []
    others = []
    for p in products:
        pn = (p.get("product_name") or "").lower()
        sn = (p.get("search_field") or "").lower()
        if norm_query in pn or norm_query in sn:
            exact.append(p)
        else:
            others.append(p)
    return exact + others



@app.route("/api/mall_search", methods=["POST"])
@log_api_call
def mall_search_api():
    try:
        data = json.loads(request.get_data().decode("utf-8"))
        
        # === [NEW] Save original user query, apply stopword and stemming normalization ===
        q_raw = data.get('query', '')
        q_norm = preprocess_query(q_raw)
        q_nostop = remove_stopwords(q_norm)
        q_tokens = [stem_rule(t) for t in q_nostop.split()]
        data['query'] = " ".join(q_tokens)
        # Save q_raw for later use in exact match boosting
        orig_user_query = q_raw

        size = data.get("size", 26)
        from_index = data.get("from_index", data.get("fromIndex", 0))
        try:
            size = int(size)
        except Exception:
            size = 26
        try:
            from_index = int(from_index)
        except Exception:
            from_index = 0
        logging.info(f"Received from_index: {from_index}, size: {size}")

        city_id = data.get("city_id", data.get("cityId", None))
        city_id = f"citi_id_{city_id}"
        sort_by = data.get("sortBy", {})

        # === Resolve Query/Filters/Category ===
        query, filters, mapped_category = resolve_query_and_filters(data, es)

        chip = data.get("chip", "").strip()
        base_category = data.get("base_category", "").strip()
        
        # === 1. Calculate Global EMI Min/Max (before applying any EMI filter!) ===
        filters_for_emi_agg = dict(filters)
        emi_filter_for_agg = filters_for_emi_agg.pop("emi", None)
        es_query_for_emi_agg = build_advanced_search_query(
            query, filters_for_emi_agg, city_id, mapped_category, price_filter=None, emi_range=None
        )
        agg_query = {
            "query": es_query_for_emi_agg,
            "aggs": {
                "min_emi": {"min": {"field": "lowest_emi"}},
                "max_emi": {"max": {"field": "lowest_emi"}},
            },
            "size": 0
        }
        try:
            agg_response = es.search(
                index=PRODUCT_INDEX_NAME,
                body=agg_query,
            )
            emi_slider_min = int(agg_response["aggregations"]["min_emi"]["value"] or 0)
            emi_slider_max = int(agg_response["aggregations"]["max_emi"]["value"] or 0)
        except Exception as agg_ex:
            logging.warning(f"EMI agg failed: {agg_ex}")
            emi_slider_min, emi_slider_max = 0, 0

        # Always combine query + chip (for suggestions & audit)
        combined_query = f"{query} {chip}".strip() if chip else query

        # === Correction and Price/EMI Parsing ===
        corrected_query_raw = correct_query(query)
        corrected_query, price_filter = parse_price_from_query(corrected_query_raw)

        #corrected_query, price_filter = parse_price_from_query(corrected_query_raw)
        chip_corrected = chip  # (hook for chip spell correction)

        corrected_combined_query = f"{corrected_query} {chip_corrected}".strip() if chip else corrected_query

        query_n = normalize(corrected_query)

        show_apple_redirect = False
        if is_apple_only_query(corrected_query):
            corrected_query = "smartphone"
            query_n = "smartphone"
            chip = ""
            show_apple_redirect = True

        if not mapped_category:
            mapped_category = CATEGORY_CANONICAL.get(query_n) or CATEGORY_CANONICAL.get(query_n.replace(" ", ""))

        # Scooter synonym normalization
        if query_n in SCOOTER_SYNONYMS:
            query_n = "two-wheeler"
            corrected_query = "two-wheeler"
            corrected_combined_query = f"{corrected_query} {chip}".strip() if chip else corrected_query

        if not mapped_category:
            mapped_category = CATEGORY_CANONICAL.get(query_n) or CATEGORY_CANONICAL.get(query_n.replace(" ", ""))

        emi_filter = filters.pop("emi", None)
        emi_range = None
        if emi_filter:
            emi_range = {}
            try:
                if isinstance(emi_filter, str):
                    import re
                    s = emi_filter.replace(' ', '').replace(',', '-')
                    match = re.match(r"(\d+)[^\d]+(\d+)", s)
                    if match:
                        emi_min, emi_max = int(match.group(1)), int(match.group(2))
                        if emi_min > emi_max:
                            emi_min, emi_max = emi_max, emi_min
                        emi_range["gte"] = emi_min
                        emi_range["lte"] = emi_max
                    else:
                        try:
                            val = int(s)
                            emi_range["gte"] = emi_range["lte"] = val
                        except Exception:
                            pass
                elif isinstance(emi_filter, dict):
                    emi_min = emi_filter.get("min") or emi_filter.get("gte")
                    emi_max = emi_filter.get("max") or emi_filter.get("lte")
                    if emi_min is not None and emi_max is not None and emi_min > emi_max:
                        emi_min, emi_max = emi_max, emi_min
                    if emi_min is not None and emi_min >= 0:
                        emi_range["gte"] = emi_min
                    if emi_max is not None and emi_max >= 0:
                        emi_range["lte"] = emi_max
            except Exception as e:
                logging.warning(f"Invalid EMI filter input: {emi_filter} ({e})")
        elif "emi_min" in filters or "emi_max" in filters:
            emi_range = {}
            emi_min = filters.pop("emi_min", None)
            emi_max = filters.pop("emi_max", None)
            if emi_min is not None:
                emi_range["gte"] = emi_min
            if emi_max is not None:
                emi_range["lte"] = emi_max

        # === Build Elasticsearch Query ===
        es_query = build_advanced_search_query(
            corrected_query, filters, city_id, mapped_category, price_filter, emi_range=emi_range
        )
        query_body = {"query": es_query}

        if sort_by and "by" in sort_by:
            order = sort_by.get("order", "asc")
            if sort_by["by"] == "emi":
                query_body["sort"] = [{"lowest_emi": {"order": order}}]
            elif sort_by["by"] == "price":
                query_body["sort"] = [{"offer_price": {"order": order}}]
        else:
            query_body["sort"] = [{"lowest_emi": {"order": "asc"}}]

        response = es.search(
            index=PRODUCT_INDEX_NAME,
            body=query_body,
            from_=from_index,
            size=size
        )

        hits = response["hits"]["hits"]
        total = response["hits"]["total"]["value"]
        
        logging.info(f"User query: {query}")
        logging.info(f"Elasticsearch query payload: {json.dumps(query_body)}")
        logging.info(f"ES Response count: {len(hits)}")
        logging.info(f"Sample product: {json.dumps(hits[0], default=str) if hits else 'No hits'}")

        resp = process_response(hits, total)
        resp = clean_products_for_plp(resp)
        logging.info(f"Full API response: {json.dumps(resp, default=str)}")
        resp["data"]["PostV1Productlist"]["data"]["emi_slider_range"] = {
            "min": emi_slider_min,
            "max": emi_slider_max
        }

        # ------- Intent-based SKU sorting (all categories) -------
        try:
            user_query = combined_query
            products_list = resp["data"]["PostV1Productlist"]["data"]["products"]
            if products_list:
                sorted_products = enhanced_sort_products(products_list, user_query)
                boosted_products = boost_exact_matches(sorted_products, data.get("query", user_query))
                # === [NEW] Now boost by exact query string match (normalized) ===
                boosted_products = rank_exact_query_first(boosted_products, orig_user_query)
                resp["data"]["PostV1Productlist"]["data"]["products"] = boosted_products
        except Exception as e:
            logging.error(f"Enhanced intent sorting failed: {e}")

        # --- Apple brand filtering for mobile/phone/apple/fon/ios queries (robust) ---
        def is_query_mobile_or_apple(q):
            mobile_terms = set(BUSINESS_SYNONYMS.get("mobiles", [])) | {
                "mobile phones", "mobiles", "smartphone", "phone", "phones", "fon", "fones", "cellphone", "cell phone"
            }
            apple_terms = set(BUSINESS_SYNONYMS.get("apple", [])) | {
                "apple", "iphone", "iphones", "ios", "apple phone", "apple phones", "apple mobile", "apple mobiles"
            }
            qn = q.lower().replace(" ", "")
            for term in mobile_terms | apple_terms:
                if term.replace(" ", "") in qn:
                    return True
            if any(x in qn for x in ["apple", "iphone", "ios", "mobile", "smartphone", "phone", "fon", "fones"]):
                return True
            return False

        if is_query_mobile_or_apple(query_n):
            filtered = []
            for prod in resp["data"]["PostV1Productlist"]["data"]["products"]:
                brand = (prod.get("manufacturer_desc") or "").strip().lower()
                name = (prod.get("product_name") or "").lower()
                asset_cat = (prod.get("asset_category_name") or "").lower()
                if (
                    brand == "apple"
                    or "apple" in name or "iphone" in name or "ios" in name
                    or "apple" in asset_cat or "iphone" in asset_cat or "ios" in asset_cat
                ):
                    continue
                filtered.append(prod)
            resp["data"]["PostV1Productlist"]["data"]["products"] = filtered
            filters_resp = resp["data"]["PostV1Productlist"]["data"]["filters"]
            for f in filters_resp:
                if "attributes" in f:
                    for attr_name in f["attributes"]:
                        before = f["attributes"][attr_name]
                        f["attributes"][attr_name] = [
                            v for v in before if not any(
                                bad in (v["name"] or "").lower() for bad in ["apple", "iphone", "ios"]
                            )
                        ]

        # --- Mobile brand sorting ---
        elif is_mobile_query(query_n):
            priority_brands = ["Samsung", "Oppo", "Vivo", "Realme", "Redmi", "Nothing"]
            buckets = {b: [] for b in priority_brands}
            others = []
            products = resp["data"]["PostV1Productlist"]["data"]["products"]
            for prod in products:
                brand = (prod.get("manufacturer_desc") or "").strip()
                brand_cmp = brand.lower()
                matched = False
                for pb in priority_brands:
                    if brand_cmp == pb.lower():
                        buckets[pb].append(prod)
                        matched = True
                        break
                if not matched:
                    others.append(prod)
            ordered_products = []
            for pb in priority_brands:
                ordered_products.extend(buckets[pb])
            ordered_products.extend(sorted(others, key=lambda p: (p.get("manufacturer_desc") or "").lower()))
            resp["data"]["PostV1Productlist"]["data"]["products"] = ordered_products

        # --- Set suggested_search_keyword, original_query, query_corrected ---
        combined_query_normalized = normalize(combined_query)
        corrected_combined_normalized = normalize(corrected_combined_query)
        query_was_corrected = (combined_query_normalized != corrected_combined_normalized)

        resp["data"]["PostV1Productlist"]["data"]["original_query"] = combined_query
        resp["data"]["PostV1Productlist"]["data"]["suggested_search_keyword"] = (
            corrected_combined_query if query_was_corrected else combined_query
        )
        resp["data"]["PostV1Productlist"]["data"]["query_corrected"] = query_was_corrected

        if show_apple_redirect:
            resp["data"]["PostV1Productlist"]["data"]["info_message"] = (
                "No Apple products found. Showing top smartphones from OnePlus, Nothing, Oppo, Samsung, and more."
            )

        return jsonify(resp), 200

    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"message": "Internal server error", "error": str(e)}), 500
    







# ========== BUSINESS AUTOSUGGEST HANDLER ==========
import re

def normalize(text):
    """Lowercase, remove special chars, and normalize spaces."""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]+', '', text.lower())).strip()

def patch_brand_token_must(parsed, user_query):
    tokens = [t for t in parsed["tokens"]]
    brands = set([b.lower() for b in parsed["brands"]])
    must = []
    must_not = []
    # Handle spacing variants for numbers/GB/MB etc.
    def gen_variants(t):
        if "gb" in t.lower() and not " " in t.lower():
            base = t.lower().replace("gb", "")
            return [t, f"{base} GB", f"{base} gb", f"{base}GB", f"{base}Gb"]
        return [t]
    if brands:
        for b in brands:
            must.append({
                "bool": {
                    "should": [
                        {"match_phrase": {"product_name": b}},
                        {"match_phrase": {"search_field": b}},
                    ],
                    "minimum_should_match": 1
                }
            })
        # Each *other* token: allow ANY variant match in any key field
        for t in tokens:
            if t.lower() not in brands and len(t) > 2:
                variants = gen_variants(t)
                must.append({
                    "bool": {
                        "should": [
                            {"multi_match": {
                                "query": variant,
                                "fields": [
                                    "product_name^3", "search_field^2", "sku", "products.name^2", "products.sku"
                                ],
                                "operator": "or",
                                "fuzziness": "AUTO"
                            }} for variant in variants
                        ],
                        "minimum_should_match": 1
                    }
                })
    return must, must_not





def get_business_autosuggestions(query):
    n_query = normalize(query)
    suggestions = set()

    # Exact match
    if n_query in BUSINESS_AUTOSUGGEST:
        suggestions.update(BUSINESS_AUTOSUGGEST[n_query])
    # Partial/startswith match
    for key in BUSINESS_AUTOSUGGEST:
        if n_query and (n_query in key or key in n_query):
            suggestions.update(BUSINESS_AUTOSUGGEST[key])

    # Enhanced: Partial/startswith matching from BUSINESS_SYNONYMS
    for synlist in BUSINESS_SYNONYMS.values():
        for syn in synlist:
            syn_norm = normalize(syn)
            if n_query and (n_query in syn_norm or syn_norm.startswith(n_query)):
                suggestions.add(syn)
    return list(dict.fromkeys(suggestions))

def get_synonym_suggestions(query):
    n_query = normalize(query)
    close_syns = []
    for canon, syns in BUSINESS_SYNONYMS.items():
        canon_norm = normalize(canon)
        if n_query and (n_query in canon_norm or canon_norm.startswith(n_query)):
            close_syns.append(canon)
        for syn in syns:
            syn_norm = normalize(syn)
            if n_query and (n_query in syn_norm or syn_norm.startswith(n_query)):
                close_syns.append(syn)
    # Deduplicate, prioritize shorter matches, return top 5
    close_syns = sorted(set(close_syns), key=len)
    return close_syns[:5]

@lru_cache(maxsize=4096)
def _cached_autosuggest_es(query_terms_tuple):
    expanded_terms = query_terms_tuple
    should_clauses = []

    PRIORITY_BRANDS = ["Samsung", "Apple", "Oppo", "Vivo", "Realme", "Redmi", "Nothing"]
    query_string = " ".join(expanded_terms).lower()

    # Stricter brand boosting
    for brand in PRIORITY_BRANDS:
        if query_string.startswith(brand.lower()) or brand.lower().startswith(query_string):
            should_clauses.append({"term": {"value.keyword": {"value": brand, "boost": 50}}})

    # Stricter synonym & canonical expansion
    for term in expanded_terms:
        if len(term) < 2: continue
        should_clauses += [
            {"match_phrase_prefix": {"value": {"query": term, "boost": 7}}},
            {"match": {"value": {"query": term, "fuzziness": "AUTO", "boost": 4}}},
        ]
    # Add canonical category boost for exact synonym match
    for cat, syns in BUSINESS_SYNONYMS.items():
        for s in syns:
            if query_string == s.lower():
                should_clauses.append({"term": {"value.keyword": {"value": CATEGORY_CANONICAL.get(cat, cat), "boost": 25}}})

    es_query = {
        "bool": {
            "should": should_clauses,
            "minimum_should_match": 1
        }
    }
    response = es.search(
        index=AUTOSUGGEST_INDEX_NAME,
        body={
            "query": es_query,
            "_source": ["value"],
            "size": 50
        }
    )
    return response

def get_enhanced_intent(query):
    colors = ["black", "white", "silver", "blue", "red", "grey", "gold", "green", "yellow", "pink", "purple"]
    storages = ["256 gb", "512 gb", "128 gb", "64 gb", "32 gb", "16 gb", "8 gb", "4 gb", "1 tb", "2 tb"]
    sizes = ["43 inch", "50 inch", "55 inch", "65 inch", "32 inch", "40 inch", "1.5 ton", "2 ton", 
             "6 kg", "7 kg", "8 kg", "9 kg", "10 kg"]
    variants = ["pro", "max", "plus", "mini", "air", "smart", "ultra", "suv", "sedan", "hatchback"]
    query = query.lower()
 
    intent = {
        "colors": [color for color in colors if color in query],
        "storages": [storage for storage in storages if storage in query],
        "sizes": [size for size in sizes if size in query],
        "variants": [variant for variant in variants if variant in query],
    }
    return intent
 
def robust_extract_attribute(prod, keys):
    for key in keys:
        value = prod.get(key, "")
        if not value:
            continue
        # If value is a dict, try extracting 'name'
        if isinstance(value, dict):
            value = value.get("name", "") or value.get("value", "") or str(value)
        # If value is a list, try extracting from first element
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                value = value[0].get("name", "") or value[0].get("value", "") or str(value[0])
            elif value and isinstance(value[0], str):
                value = value[0]
            else:
                value = str(value)
        # Now, make sure it's a string
        if isinstance(value, str):
            return value.lower()
        else:
            return str(value).lower()
    # Fallback for nested products
    if "products" in prod and prod["products"]:
        for sku in prod["products"]:
            out = robust_extract_attribute(sku, keys)
            if out:
                return out
    return ""

 
def enhanced_score_product(prod, intent):
    score = 0
 
    prod_color = robust_extract_attribute(prod, ["color", "attribute_color", "attribute_swatch_color"])
    prod_storage = robust_extract_attribute(prod, ["storage", "attribute_storage", "attribute_internal_storage"])
    prod_size = robust_extract_attribute(prod, ["size", "screen_size", "attribute_screen_size_in_inches"])
    prod_variant = robust_extract_attribute(prod, ["model", "variant", "attribute_variant", 
                                                   "category_type", "asset_category_name"])
 
    # Dynamic scoring based on user intent
    if intent["colors"]:
        if any(c in prod_color for c in intent["colors"]):
            score += 5
    if intent["storages"]:
        if any(s in prod_storage for s in intent["storages"]):
            score += 4
    if intent["sizes"]:
        if any(sz in prod_size for sz in intent["sizes"]):
            score += 3
    if intent["variants"]:
        if any(v in prod_variant for v in intent["variants"]):
            score += 2
 
    # Negative for descending sort (highest score first)
    return -score
 
def enhanced_sort_products(products, query):
    intent = get_enhanced_intent(query)
    return sorted(products, key=lambda p: enhanced_score_product(p, intent))



@app.route("/api/mall_autosuggest", methods=["POST"])
@log_api_call
def mall_autosuggest_api():
    try:
        start_time = time.time()

        data = json.loads(request.get_data().decode("utf-8"))
        query = data.get("query", "")
        if not query or not query.strip():
            return jsonify({"error": "Query parameter 'query' is required"}), 400

        corrected_query = correct_query(query)
        expanded_terms = tuple(expand_search_terms(corrected_query))

        # BUSINESS SUGGESTIONS FIRST
        business_suggestions = get_business_autosuggestions(corrected_query)

        # SYNONYM SUGGESTIONS (NEW)
        synonym_suggestions = get_synonym_suggestions(corrected_query)

        # THEN ES SUGGESTIONS
        es_start = time.time()
        response = _cached_autosuggest_es(expanded_terms)
        es_end = time.time()

        variant_map = defaultdict(set)
        for hit in response["hits"]["hits"]:
            val = hit["_source"]["value"].title().strip()
            # Defensive: split on commas if the value accidentally contains multiple products
            suggestions = [v.strip() for v in val.split(",") if v.strip()]
            for sug in suggestions:
                base = re.sub(r"\s*\(.*\)$", "", sug)
                variant = re.search(r"\((.*?)\)$", sug)
                variant_name = variant.group(1) if variant else ""
                variant_map[base].add(variant_name)


        PRIORITY_BRANDS = ["Samsung", "Apple", "Oppo", "Vivo", "Realme", "Redmi", "Nothing"]
        keywords = []
        query_lower = corrected_query.lower()
        for brand in PRIORITY_BRANDS:
            if any(base.lower().startswith(brand.lower()) or brand.lower().startswith(query_lower)
                   for base in variant_map.keys()):
                for base in variant_map.keys():
                    if base.lower().startswith(brand.lower()):
                        n_var = len([v for v in variant_map[base] if v])
                        if n_var > 1:
                            keywords.append(f"{base} (+{n_var} variants)")
                        else:
                            keywords.append(base)
        already = set([k.split(" (+")[0] for k in keywords])
        for base, variants in variant_map.items():
            if base in already:
                continue
            n_var = len([v for v in variants if v])
            if n_var > 1:
                keywords.append(f"{base} (+{n_var} variants)")
            else:
                keywords.append(base)
        keywords = keywords[:15]

        # Merge business and ES results: business always first, deduped
        all_keywords = list(dict.fromkeys(business_suggestions + keywords))
        all_keywords = all_keywords[:15]

        # ===== Chips/filter_text logic (robust for synonyms and major categories) =====
        q_lower = normalize(corrected_query)
        mapped_category = CATEGORY_CANONICAL.get(q_lower)
        if not mapped_category:
            for cat_key, syns in BUSINESS_SYNONYMS.items():
                if q_lower in [normalize(s) for s in syns]:
                    mapped_category = cat_key
                    break

        # ========== ENHANCED CHIPS & FILTER LOGIC ==========
        chips, filter_text = [], ""
        q_lower = normalize(corrected_query)
        mapped_category = CATEGORY_CANONICAL.get(q_lower)

        # 1. Try direct mapping (canonical)
        if mapped_category and mapped_category in CATEGORY_HARDCODED_CHIPS:
            chips, filter_text = CATEGORY_HARDCODED_CHIPS[mapped_category]

        # 2. If not found, try synonyms (BUSINESS_SYNONYMS)
        elif not chips:
            for cat_key, syns in BUSINESS_SYNONYMS.items():
                # Check normalized synonyms
                if q_lower in [normalize(s) for s in syns]:
                    if cat_key in CATEGORY_HARDCODED_CHIPS:
                        chips, filter_text = CATEGORY_HARDCODED_CHIPS[cat_key]
                        break

        # 3. If still not found, try business autosuggest mapping
        elif not chips:
            for key in BUSINESS_AUTOSUGGEST:
                if q_lower == key or q_lower in key or key in q_lower:
                    # Suggest chips for business intent if mapped to a known category
                    cat_map = CATEGORY_CANONICAL.get(key)
                    if cat_map and cat_map in CATEGORY_HARDCODED_CHIPS:
                        chips, filter_text = CATEGORY_HARDCODED_CHIPS[cat_map]
                        break

        # 4. Fallback for mobiles (most common use-case)
        if not chips and q_lower in BUSINESS_SYNONYMS.get("mobiles", []):
            chips, filter_text = CATEGORY_HARDCODED_CHIPS["mobiles"]

        total_time = (time.time() - start_time) * 1000
        es_time = (es_end - es_start) * 1000
        print(f"[Autosuggest Timing] ES: {es_time:.2f} ms | Total: {total_time:.2f} ms | Query: {query}")

        out = {
            "message": "no autosuggest issue",
            "response": {
                "chips": chips,
                "filter_text": filter_text,
                "keywords": all_keywords,
                "synonym_suggestions": synonym_suggestions,  # <--- ADDED!
                "language": "english"
            }
        }
        return jsonify(out), 200

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"message": "Internal server error", "error": str(e)}), 500





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8007)
