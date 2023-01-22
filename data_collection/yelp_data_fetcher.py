import requests

api_key = 'alM3qIinDSvQXVBmewGAN-B6AismyOqVpcTYmpYuXeIADKvq8UXnWcYYAHEGqaKEdEpzZ_Kmo0m-jp2iLXvk2aAA2NNbR1-XbShXl5OO98A29pQxD-yb4yHo7qPIY3Yx'

location = 'San Francisco'
headers = {
    'Authorization': f'Bearer {api_key}'
}
limit = 1000

def get_all_reviews_of_business(business_id):
    offset = 0
    while True:
        url = f'https://api.yelp.com/v3/businesses/{business_id}/reviews?offset={offset}'
        response = requests.get(url, headers=headers)
        data = response.json()
        #TODO(azheng): add review info to database
        reviews = data['reviews']
        review_offset += reviews_per_call
        if len(reviews) < reviews_per_call: return

def get_all_business_info(location):
    offset = 0
    while True:
        url = f'https://api.yelp.com/v3/businesses/search?location={location}&limit={limit}&offset={offset}'
        response = requests.get(url, headers=headers)
        data = response.json()
        businesses = data['businesses']
        offset += limit
        for business in businesses:
            if business['is_closed']: continue
            business_id = business['id']
            # TODO(azheng): add information of business, namely tags and name of business, into database.
            get_all_reviews_of_business(business_id)
        if len(businesses) < limit: return
