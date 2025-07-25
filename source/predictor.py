from prophet import Prophet
import pandas as pd


class ProphetPredictor:
    def __init__(self,  **model_params):
        """
        Инициализация класса с возможностью передачи параметров Prophet.
        """
        self.model_params = model_params
        self.model = None

    @staticmethod
    def get_season(date):
        if date.month in [12, 1, 2]:
            return 'winter'
        elif date.month in [3, 4, 5]:
            return 'spring'
        elif date.month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'

    def preprocess(self, df):
        df['weeks_to_newyear'] = df['ds'].apply(lambda d: (
            pd.Timestamp(year=d.year, month=1, day=1) - d).days // 7)
        df['season'] = df['ds'].apply(self.get_season)
        # One-hot кодировка столбцов
        df = pd.concat([df, pd.get_dummies(df['season'])], axis=1)
        # Добавление праздников
        years = range(df['ds'].dt.year.min(), df['ds'].dt.year.max() + 1)
        self.holidays = pd.DataFrame({
            'holiday': ['Новый год'] * len(years) + ['День Защитника Отечества'] * len(years) + ['Международный женский день'] * len(years),
            'ds': pd.to_datetime([f'{y}-01-01' for y in years] + [f'{y}-02-23' for y in years] + [f'{y}-03-08' for y in years]),
            'lower_window': -14,  # за сколько дней до праздника учитывать влияние
            'upper_window': 7  # сколько дней после праздника учитывать влияние
        })
        return df

    def add_future_regressors(self, future_df):
        future_df['weeks_to_newyear'] = future_df['ds'].apply(
            lambda d: (pd.Timestamp(year=d.year, month=1, day=1) - d).days // 7)
        future_df['season'] = future_df['ds'].apply(self.get_season)
        dummies = pd.get_dummies(future_df['season'])
        for season in ['winter', 'spring', 'summer', 'autumn']:
            if season not in dummies:
                dummies[season] = 0  # если сезон не встречался — добавить
        future_df = pd.concat(
            [future_df, dummies[['winter', 'spring', 'summer', 'autumn']]], axis=1)
        return future_df

    def fit(self, df):
        """
        Обучает модель Prophet на переданном датасете.
        df: pandas.DataFrame с колонками 'ds' (дата) и 'y' (значение)
        """
        df = self.preprocess(df)
        self.model = Prophet(holidays=self.holidays, **self.model_params)
        self.model.add_regressor('weeks_to_newyear')
        for season in ['winter', 'spring', 'summer', 'autumn']:
            self.model.add_regressor(season)
        self.model.fit(df)

    def predict(self, future_df):
        """
        Получает прогноз модели на df, содержащем колонку 'ds' с будущими датами.
        Возвращает датафрейм с прогнозом.
        """
        if self.model is None:
            raise Exception("Сначала вызовите метод fit().")
        # Добавляем спец. регрессоры
        future_df = self.add_future_regressors(future_df)
        forecast = self.model.predict(future_df)
        return forecast

    def fit_predict(self, train_df, periods, freq='W'):
        """
        Обучает модель и сразу возвращает прогноз на указанное количество периодов.
        train_df: обучающий DataFrame с 'ds' и 'y'
        periods: сколько периодов вперёд прогнозировать
        freq: частота ('W' - недели, 'M' — месяцы)
        """
        self.fit(train_df)
        last_date = train_df['ds'].max()
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        return self.predict(future)

    def save_predict_plot(self, last_k=7*4, ):
        pass
