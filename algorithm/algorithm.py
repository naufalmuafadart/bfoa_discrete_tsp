import pandas as pd

class Algorithm:
    def __init__(self, ids, hotel_id, doi_duration, doi_cost, doi_rating, days_count, n):
        self.PREFERENCE_ID = list(map(int, ids))  # => [1,2,3]
        self.HOTEL_ID = hotel_id
        self.DOI_DURATION = doi_duration
        self.DOI_COST = doi_cost
        self.DOI_RATING = doi_rating
        self.DOI_POI_INCLUDED = doi_rating
        self.DAYS_COUNT = days_count
        self.AGENT_LENGTH = len(self.PREFERENCE_ID)
        self.DEPART_TIME = 8 * 3600
        self.ARRIVAL_TIME = 21 * 3600
        self.n = n

        # get dataset
        self.prepare_dataset()

    def prepare_dataset(self):
        self.df_places = pd.read_csv('./dataset/places.csv')
        self.df_time_matrix = pd.read_csv('./dataset/place_timematrix.csv')
        self.df_schedule = pd.read_csv('./dataset/place_jadwal.csv')
        self.df_schedule['jam_buka'] = self.df_schedule['jam_buka'].apply(lambda x: int(x[:2]) * 3600 + int(x[3:]) * 60)
        self.df_schedule['jam_tutup'] = self.df_schedule['jam_tutup'].apply(lambda x: int(x[:2]) * 3600 + int(x[3:]) * 60)

    def get_travel_time(self, a, b):
        df_filter = self.df_time_matrix[(self.df_time_matrix['id_a'] == a) & (self.df_time_matrix['id_b'] == b)]
        if df_filter.empty:
            raise IndexError('Index POI not found')
        return df_filter.iloc[0]['durasi']

    def tsp_fitness_function(self, solution):
        if solution is None or solution == [] or not isinstance(solution, list):
            raise ValueError('Solution not valid')
        if len(set(solution)) != len(solution):
            raise ValueError('Solution has duplicate value')
        if len(solution) != self.AGENT_LENGTH:
            raise ValueError('Solution length is not correct')
        duration = 0
        for index, poi in enumerate(solution):
            if index == 0:
                duration += self.get_travel_time(self.HOTEL_ID, poi)
                if len(solution) == 1:
                    duration += self.get_travel_time(poi, self.HOTEL_ID)
            elif index == len(solution) - 1:
                duration += self.get_travel_time(solution[index - 1], poi)
                duration += self.get_travel_time(poi, self.HOTEL_ID)
            else:
                duration += self.get_travel_time(solution[index - 1], poi)
        return duration
