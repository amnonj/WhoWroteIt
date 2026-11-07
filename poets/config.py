import pandas as pd
import datetime

CSV_FILENAME = 'poetry_data/poetry_dataset.csv'
MIN_POST_LENGTH = 50
AUTHOR_LIST = ["חיים נחמן ביאליק","שאול טשרניחובסקי","רחל בלובשטיין","אברהם שלונסקי","דן אלמגור",
                "יהודה הלוי","שלמה אבן גבירול","רחל שפירא"]
AUTHORS_CODES = (89, 57, 141, 372, 450, 161, 180, 577)
pd.set_option("display.max_columns", None)

# Used to filter non poet works
MIN_NIKUD_RATIO_PER_WORK = 0.05
MAX_MEAN_LINE_LENGTH = 200
# Used to remove note line before and after the song
MIN_NIKUD_RATIO_PER_LINE = 0.20

MAX_CONSECUTIVE_LF = 2