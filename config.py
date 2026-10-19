import pandas as pd
import datetime

POSTS_URL = 'https://github.com/jschler/ILElectionData/raw/refs/heads/main/dataset/posts/AllPosts201811_202003.csv.zip'
CSV_FILENAME = 'AllPosts201811_202003.csv'
MIN_POST_LENGTH = 50
AUTHOR_LIST = ["Avigdor Liberman", "Benjamin Netanyahu", "Bezazel Smotrich", "Itamar Ben-Gvir", "Miri Regev",
               "Naftali Bennett", "Yair Lapid"]
VALIDATION_START = datetime.datetime(2019, 12, 1)
TEST_START = datetime.datetime(2020, 2, 1)
pd.set_option("display.max_columns", None)

PERSON_NAMES = (
    # Benjamin Netanyahu
    "בנימין", "נתניהו", "ביבי",
    "Benjamin", "Binyamin", "Netanyahu", "Bibi",

    # Yair Lapid
    "יאיר", "לפיד", "לפידקאסט",
    "Yair", "Lapid",

    # Naftali Bennett
    "נפתלי", "בנט", "בנטים",
    "Naftali", "Bennett",

    # Avigdor Liberman
    "אביגדור", "ליברמן",
    "Avigdor", "Liberman", "Lieberman",

    # Bezalel Smotrich
    "בצלאל", "סמוטריץ", "סמוטריץ'",
    "Bezalel", "Bezazel", "Smotrich",

    # Itamar Ben-Gvir
    "איתמר", "בן גביר",
    "Itamar", "Ben Gvir", "Bengvir",

    # Miri Regev
    "מירי", "רגב",
    "Miri", "Regev"
)

PARTY_NAMES = (
    # Likud
    "הליכוד", "ליכוד",
    "Likud",

    # Yesh Atid
    "יש עתיד",
    "Yesh Atid",

    # Bennett and Lapid - newer parties
    "בנט 2026",
    "ביחד",
    "Bennett 2026",
    "Beyachad",

    # Bennett - historical parties
    "הבית היהודי",
    "הימין החדש",
    "ימינה",
    "הרוח הציונית",
    "Jewish Home",
    "The Jewish Home",
    "New Right",
    "The New Right",
    "Yamina",
    "Zionist Spirit",

    # Yisrael Beiteinu
    "ישראל ביתנו",
    "Yisrael Beiteinu",
    "Yisrael Beytenu",
    "Israel Beiteinu",
    "Israel Beytenu",

    # Religious Zionism / Otzma Yehudit
    "האיחוד הלאומי",
    "תקומה",
    "איחוד מפלגות הימין",
    "הציונות הדתית",
    "עוצמה יהודית",
    "נעם",
    "National Union",
    "Tkuma",
    "Union of Right-Wing Parties",
    "Religious Zionism",
    "Otzma Yehudit",
    "Jewish Power",
    "Noam"
)

ROLE_NAMES = (
    "ראש הממשלה",
    "ראש ממשלה",
    "ראש ממשלת ישראל",
    "ראש ממשלת",
    'רה"מ',
    "Prime Minister",
    "Deputy Prime Minister",
    "ראש האופוזיציה",
    "יושב ראש האופוזיציה",
    'יו"ר האופוזיציה',
    "יושב ראש הכנסת",
    'יו"ר הכנסת',
    "שר הביטחון",
    "שרת הביטחון",
    "שר הבטחון",
    "שרת הבטחון",
    "שר התחבורה",
    "שרת התחבורה",
    "שר החינוך",
    "שרת החינוך",
    "שר האוצר",
    "שרת האוצר",
    "שר הפנים",
    "שרת הפנים",
    "שר התרבות והספורט",
    "שרת התרבות והספורט",
    "שר המשפטים",
    "שרת המשפטים",
    "שר החוץ",
    "שרת החוץ",
    "שר הבריאות",
    "שרת הבריאות"
)

PREFIXES = ("וכשה", "וכש", "כשה", "כש", "ושב", "ושל", "ושמ", "ושה", "שב", "שכ", "של", "שמ",
            "שה", "וב", "וכ", "ול", "ומ", "וש", "וה", "מה", "ב", "כ", "ל", "מ", "ו", "ש", "ה")
