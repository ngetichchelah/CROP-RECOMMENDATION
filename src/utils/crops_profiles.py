"""
Crop profiles with economic, sustainability, and descriptive insights
"""

CROP_PROFILES = {
    'rice': {
        'economic_value': 'Medium',
        'water_efficiency': 'Low',
        'market_demand': 'Very High',
        'sustainability': 'Medium',
        'labor_intensity': 'High',
        'season': 'Monsoon',
        'typical_yield': '4-5 tons/ha',
        'market_price': '$400-600/ton',
        'description': 'Rice thrives in flooded or monsoon conditions and remains a major staple with strong global demand.'
    },
    'maize': {
        'economic_value': 'Medium',
        'water_efficiency': 'Medium',
        'market_demand': 'High',
        'sustainability': 'High',
        'labor_intensity': 'Medium',
        'season': 'Spring/Summer',
        'typical_yield': '5-7 tons/ha',
        'market_price': '$200-300/ton',
        'description': 'Maize adapts to diverse soils and climates, offering steady yields and high market relevance as a staple crop.'
    },
    'coffee': {
        'economic_value': 'Very High',
        'water_efficiency': 'Medium',
        'market_demand': 'Very High',
        'sustainability': 'Medium',
        'labor_intensity': 'High',
        'season': 'Year-round',
        'typical_yield': '1-2 tons/ha',
        'market_price': '$3,000-5,000/ton',
        'description': 'Coffee is a high-value perennial requiring shaded, well-drained soils and careful post-harvest handling.'
    },
    'chickpea': {
        'economic_value': 'Medium',
        'water_efficiency': 'Very High',
        'market_demand': 'Medium',
        'sustainability': 'Very High',
        'labor_intensity': 'Low',
        'season': 'Winter/Spring',
        'typical_yield': '1-2 tons/ha',
        'market_price': '$600-900/ton',
        'description': 'Chickpea is drought-tolerant and enriches soil fertility through nitrogen fixation, ideal for semi-arid areas.'
    },
    'cotton': {
        'economic_value': 'High',
        'water_efficiency': 'Low',
        'market_demand': 'High',
        'sustainability': 'Low',
        'labor_intensity': 'High',
        'season': 'Spring/Summer',
        'typical_yield': '2-3 tons/ha',
        'market_price': '$1,500-2,000/ton',
        'description': 'Cotton is a major cash crop valued for its fiber, though it requires intensive management and high water input.'
    },
    'banana': {
        'economic_value': 'Medium',
        'water_efficiency': 'Low',
        'market_demand': 'High',
        'sustainability': 'Medium',
        'labor_intensity': 'Medium',
        'season': 'Year-round',
        'typical_yield': '30-40 tons/ha',
        'market_price': '$300-500/ton',
        'description': 'Banana grows well in humid, warm regions and provides year-round harvest with consistent market demand.'
    },
    'watermelon': {
        'economic_value': 'Medium',
        'water_efficiency': 'Low',
        'market_demand': 'High',
        'sustainability': 'Medium',
        'labor_intensity': 'Medium',
        'season': 'Summer',
        'typical_yield': '20-30 tons/ha',
        'market_price': '$200-400/ton',
        'description': 'Watermelon is a fast-growing summer fruit that thrives in warm temperatures and sandy, well-drained soils.'
    },
    'grapes': {
        'economic_value': 'High',
        'water_efficiency': 'Medium',
        'market_demand': 'High',
        'sustainability': 'Medium',
        'labor_intensity': 'Very High',
        'season': 'Spring/Summer',
        'typical_yield': '10-15 tons/ha',
        'market_price': '$600-1,000/ton',
        'description': 'Grapes are high-value fruits suited for sunny climates; require trellising and skilled vineyard management.'
    },
    'lentil': {
        'economic_value': 'Medium',
        'water_efficiency': 'High',
        'market_demand': 'Medium',
        'sustainability': 'Very High',
        'labor_intensity': 'Low',
        'season': 'Winter',
        'typical_yield': '1-1.5 tons/ha',
        'market_price': '$700-1,000/ton',
        'description': 'Lentils are hardy, water-efficient legumes that improve soil nitrogen and thrive in cool, dry conditions.'
    },
    'pomegranate': {
        'economic_value': 'High',
        'water_efficiency': 'Medium',
        'market_demand': 'High',
        'sustainability': 'High',
        'labor_intensity': 'Medium',
        'season': 'Spring/Fall',
        'typical_yield': '10-15 tons/ha',
        'market_price': '$800-1,200/ton',
        'description': 'Pomegranate is a drought-tolerant fruit with growing global demand and excellent export potential.'
    },
    'coconut': {
        'economic_value': 'High',
        'water_efficiency': 'Low',
        'market_demand': 'Very High',
        'sustainability': 'High',
        'labor_intensity': 'Low',
        'season': 'Year-round',
        'typical_yield': '80-100 nuts/tree',
        'market_price': '$400-600/ton',
        'description': 'Coconut palms thrive in coastal and tropical zones, providing multiple income streams from oil, fiber, and water.'
    },
    'papaya': {
        'economic_value': 'Medium',
        'water_efficiency': 'Low',
        'market_demand': 'High',
        'sustainability': 'Medium',
        'labor_intensity': 'Low',
        'season': 'Year-round',
        'typical_yield': '40-60 tons/ha',
        'market_price': '$300-500/ton',
        'description': 'Papaya grows quickly in tropical climates and yields continuously, offering good returns for small farmers.'
    },
    'jute': {
        'economic_value': 'Medium',
        'water_efficiency': 'Medium',
        'market_demand': 'Medium',
        'sustainability': 'High',
        'labor_intensity': 'High',
        'season': 'Monsoon',
        'typical_yield': '2-3 tons/ha',
        'market_price': '$400-600/ton',
        'description': 'Jute is a sustainable fiber crop ideal for wet monsoon regions, valued for eco-friendly textile production.'
    },
    'mango': {
        'economic_value': 'High',
        'water_efficiency': 'Medium',
        'market_demand': 'Very High',
        'sustainability': 'High',
        'labor_intensity': 'Medium',
        'season': 'Spring/Summer',
        'typical_yield': '10-15 tons/ha',
        'market_price': '$400-800/ton',
        'description': 'Mango, the “king of fruits,” thrives in warm tropical regions and commands excellent local and export markets.'
    },
    'apple': {
        'economic_value': 'High',
        'water_efficiency': 'Medium',
        'market_demand': 'Very High',
        'sustainability': 'High',
        'labor_intensity': 'High',
        'season': 'Fall',
        'typical_yield': '15-25 tons/ha',
        'market_price': '$600-1,000/ton',
        'description': 'Apples prefer temperate climates and rich soils, offering reliable profits through both fresh and processed markets.'
    },
    'orange': {
        'economic_value': 'Medium',
        'water_efficiency': 'Medium',
        'market_demand': 'High',
        'sustainability': 'High',
        'labor_intensity': 'Medium',
        'season': 'Winter',
        'typical_yield': '20-30 tons/ha',
        'market_price': '$300-500/ton',
        'description': 'Oranges are high-demand citrus fruits suited for warm, frost-free climates and support year-round juice markets.'
    },
    'muskmelon': {
        'economic_value': 'Medium',
        'water_efficiency': 'Low',
        'market_demand': 'Medium',
        'sustainability': 'Medium',
        'labor_intensity': 'Medium',
        'season': 'Summer',
        'typical_yield': '15-25 tons/ha',
        'market_price': '$200-400/ton',
        'description': 'Muskmelon matures quickly under warm, sunny conditions and offers seasonal income through fresh markets.'
    },
    'kidneybeans': {
        'economic_value': 'Medium',
        'water_efficiency': 'Medium',
        'market_demand': 'High',
        'sustainability': 'Very High',
        'labor_intensity': 'Low',
        'season': 'Spring/Fall',
        'typical_yield': '1.5-2.5 tons/ha',
        'market_price': '$800-1,200/ton',
        'description': 'Kidney beans are protein-rich legumes that fix nitrogen in the soil and thrive in moderately warm climates.'
    },
    'pigeonpeas': {
        'economic_value': 'Low',
        'water_efficiency': 'Very High',
        'market_demand': 'Medium',
        'sustainability': 'Very High',
        'labor_intensity': 'Low',
        'season': 'Monsoon/Post-monsoon',
        'typical_yield': '0.8-1.5 tons/ha',
        'market_price': '$400-600/ton',
        'description': 'Pigeon peas are deep-rooted, drought-resistant crops ideal for marginal soils and low-rainfall areas.'
    },
    'mothbeans': {
        'economic_value': 'Low',
        'water_efficiency': 'Very High',
        'market_demand': 'Low',
        'sustainability': 'Very High',
        'labor_intensity': 'Low',
        'season': 'Summer',
        'typical_yield': '0.5-1 tons/ha',
        'market_price': '$300-500/ton',
        'description': 'Moth beans are extremely drought-tolerant and suitable for drylands, improving soil structure and fertility.'
    },
    'mungbean': {
        'economic_value': 'Medium',
        'water_efficiency': 'High',
        'market_demand': 'High',
        'sustainability': 'Very High',
        'labor_intensity': 'Low',
        'season': 'Spring/Summer',
        'typical_yield': '0.8-1.2 tons/ha',
        'market_price': '$600-900/ton',
        'description': 'Mung bean grows quickly and improves soil nitrogen, providing nutritious protein under low-rainfall conditions.'
    },
    'blackgram': {
        'economic_value': 'Medium',
        'water_efficiency': 'High',
        'market_demand': 'High',
        'sustainability': 'Very High',
        'labor_intensity': 'Low',
        'season': 'Monsoon',
        'typical_yield': '0.6-1 tons/ha',
        'market_price': '$700-1,000/ton',
        'description': 'Black gram is a short-duration pulse crop that enriches soil fertility and suits monsoon-based farming systems.'
    },
    'groundnuts': {
        'economic_value': 'Medium',
        'water_efficiency': 'Medium',
        'market_demand': 'High',
        'sustainability': 'High',
        'labor_intensity': 'Medium',
        'season': 'Monsoon',
        'typical_yield': '2-3 tons/ha',
        'market_price': '$500-800/ton',
        'description': 'Groundnuts are oil-rich legumes suited for sandy soils; they improve soil fertility and have high local demand.'
    }
}


def get_crop_profile(crop_name):
    """Get profile for a specific crop"""
    return CROP_PROFILES.get(crop_name.lower(), {
        'economic_value': 'N/A',
        'water_efficiency': 'N/A',
        'market_demand': 'N/A',
        'sustainability': 'N/A',
        'typical_yield': 'N/A',
        'market_price': 'N/A',
        'labor_intensity': 'N/A',
        'description': 'No description available for this crop.'
    })


def get_all_crops():
    """Get list of all crops with profiles"""
    return list(CROP_PROFILES.keys())


__all__ = ['get_crop_profile', 'get_all_crops', 'CROP_PROFILES']


# """
# Crop profiles with economic and sustainability data
# """

# CROP_PROFILES = {
#     'rice': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'Low',
#         'market_demand': 'Very High',
#         'sustainability': 'Medium',
#         'labor_intensity': 'High',
#         'season': 'Monsoon',
#         'typical_yield': '4-5 tons/ha',
#         'market_price': '$400-600/ton'
#     },
#     'maize': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'Medium',
#         'market_demand': 'High',
#         'sustainability': 'High',
#         'labor_intensity': 'Medium',
#         'season': 'Spring/Summer',
#         'typical_yield': '5-7 tons/ha',
#         'market_price': '$200-300/ton'
#     },
#     'coffee': {
#         'economic_value': 'Very High',
#         'water_efficiency': 'Medium',
#         'market_demand': 'Very High',
#         'sustainability': 'Medium',
#         'labor_intensity': 'High',
#         'season': 'Year-round',
#         'typical_yield': '1-2 tons/ha',
#         'market_price': '$3,000-5,000/ton'
#     },
#     'chickpea': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'Very High',
#         'market_demand': 'Medium',
#         'sustainability': 'Very High',
#         'labor_intensity': 'Low',
#         'season': 'Winter/Spring',
#         'typical_yield': '1-2 tons/ha',
#         'market_price': '$600-900/ton'
#     },
#     'cotton': {
#         'economic_value': 'High',
#         'water_efficiency': 'Low',
#         'market_demand': 'High',
#         'sustainability': 'Low',
#         'labor_intensity': 'High',
#         'season': 'Spring/Summer',
#         'typical_yield': '2-3 tons/ha',
#         'market_price': '$1,500-2,000/ton'
#     },
#     'banana': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'Low',
#         'market_demand': 'High',
#         'sustainability': 'Medium',
#         'labor_intensity': 'Medium',
#         'season': 'Year-round',
#         'typical_yield': '30-40 tons/ha',
#         'market_price': '$300-500/ton'
#     },
#     'watermelon': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'Low',
#         'market_demand': 'High',
#         'sustainability': 'Medium',
#         'labor_intensity': 'Medium',
#         'season': 'Summer',
#         'typical_yield': '20-30 tons/ha',
#         'market_price': '$200-400/ton'
#     },
#     'grapes': {
#         'economic_value': 'High',
#         'water_efficiency': 'Medium',
#         'market_demand': 'High',
#         'sustainability': 'Medium',
#         'labor_intensity': 'Very High',
#         'season': 'Spring/Summer',
#         'typical_yield': '10-15 tons/ha',
#         'market_price': '$600-1,000/ton'
#     },
#     'lentil': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'High',
#         'market_demand': 'Medium',
#         'sustainability': 'Very High',
#         'labor_intensity': 'Low',
#         'season': 'Winter',
#         'typical_yield': '1-1.5 tons/ha',
#         'market_price': '$700-1,000/ton'
#     },
#     'pomegranate': {
#         'economic_value': 'High',
#         'water_efficiency': 'Medium',
#         'market_demand': 'High',
#         'sustainability': 'High',
#         'labor_intensity': 'Medium',
#         'season': 'Spring/Fall',
#         'typical_yield': '10-15 tons/ha',
#         'market_price': '$800-1,200/ton'
#     },
#     'coconut': {
#         'economic_value': 'High',
#         'water_efficiency': 'Low',
#         'market_demand': 'Very High',
#         'sustainability': 'High',
#         'labor_intensity': 'Low',
#         'season': 'Year-round',
#         'typical_yield': '80-100 nuts/tree',
#         'market_price': '$400-600/ton'
#     },
#     'papaya': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'Low',
#         'market_demand': 'High',
#         'sustainability': 'Medium',
#         'labor_intensity': 'Low',
#         'season': 'Year-round',
#         'typical_yield': '40-60 tons/ha',
#         'market_price': '$300-500/ton'
#     },
#     'jute': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'Medium',
#         'market_demand': 'Medium',
#         'sustainability': 'High',
#         'labor_intensity': 'High',
#         'season': 'Monsoon',
#         'typical_yield': '2-3 tons/ha',
#         'market_price': '$400-600/ton'
#     },
#     'mango': {
#         'economic_value': 'High',
#         'water_efficiency': 'Medium',
#         'market_demand': 'Very High',
#         'sustainability': 'High',
#         'labor_intensity': 'Medium',
#         'season': 'Spring/Summer',
#         'typical_yield': '10-15 tons/ha',
#         'market_price': '$400-800/ton'
#     },
#     'apple': {
#         'economic_value': 'High',
#         'water_efficiency': 'Medium',
#         'market_demand': 'Very High',
#         'sustainability': 'High',
#         'labor_intensity': 'High',
#         'season': 'Fall',
#         'typical_yield': '15-25 tons/ha',
#         'market_price': '$600-1,000/ton'
#     },
#     'orange': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'Medium',
#         'market_demand': 'High',
#         'sustainability': 'High',
#         'labor_intensity': 'Medium',
#         'season': 'Winter',
#         'typical_yield': '20-30 tons/ha',
#         'market_price': '$300-500/ton'
#     },
#     'muskmelon': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'Low',
#         'market_demand': 'Medium',
#         'sustainability': 'Medium',
#         'labor_intensity': 'Medium',
#         'season': 'Summer',
#         'typical_yield': '15-25 tons/ha',
#         'market_price': '$200-400/ton'
#     },
#     'kidneybeans': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'Medium',
#         'market_demand': 'High',
#         'sustainability': 'Very High',
#         'labor_intensity': 'Low',
#         'season': 'Spring/Fall',
#         'typical_yield': '1.5-2.5 tons/ha',
#         'market_price': '$800-1,200/ton'
#     },
#     'pigeonpeas': {
#         'economic_value': 'Low',
#         'water_efficiency': 'Very High',
#         'market_demand': 'Medium',
#         'sustainability': 'Very High',
#         'labor_intensity': 'Low',
#         'season': 'Monsoon/Post-monsoon',
#         'typical_yield': '0.8-1.5 tons/ha',
#         'market_price': '$400-600/ton'
#     },
#     'mothbeans': {
#         'economic_value': 'Low',
#         'water_efficiency': 'Very High',
#         'market_demand': 'Low',
#         'sustainability': 'Very High',
#         'labor_intensity': 'Low',
#         'season': 'Summer',
#         'typical_yield': '0.5-1 tons/ha',
#         'market_price': '$300-500/ton'
#     },
#     'mungbean': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'High',
#         'market_demand': 'High',
#         'sustainability': 'Very High',
#         'labor_intensity': 'Low',
#         'season': 'Spring/Summer',
#         'typical_yield': '0.8-1.2 tons/ha',
#         'market_price': '$600-900/ton'
#     },
#     'blackgram': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'High',
#         'market_demand': 'High',
#         'sustainability': 'Very High',
#         'labor_intensity': 'Low',
#         'season': 'Monsoon',
#         'typical_yield': '0.6-1 tons/ha',
#         'market_price': '$700-1,000/ton'
#     },
#     'groundnuts': {
#         'economic_value': 'Medium',
#         'water_efficiency': 'Medium',
#         'market_demand': 'High',
#         'sustainability': 'High',
#         'labor_intensity': 'Medium',
#         'season': 'Monsoon',
#         'typical_yield': '2-3 tons/ha',
#         'market_price': '$500-800/ton'
#     }
# }

# def get_crop_profile(crop_name):
#     """Get profile for a specific crop"""
#     return CROP_PROFILES.get(crop_name.lower(), {
#         'economic_value': 'N/A',
#         'water_efficiency': 'N/A',
#         'market_demand': 'N/A',
#         'sustainability': 'N/A',
#         'typical_yield': 'N/A',
#         'market_price': 'N/A',
#         'labor_intensity': 'N/A'
#     })
    
    
# def get_all_crops():
#     """Get list of all crops with profiles"""
#     return list(CROP_PROFILES.keys())

# # For easy import
# __all__ = ['get_crop_profile', 'get_all_crops', 'CROP_PROFILES']