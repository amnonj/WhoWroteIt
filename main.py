import data

filtered_posts_df = data.load_posts()

filtered_posts_df.to_csv("filtered_posts_df.csv", index=False, encoding="utf-8-sig")
