from datetime import datetime
import numpy as np

"""

Helper code to generate 10-dimensional user profile feature based on crawled user object using Twitter Developer API

"""

def hand_feature(user_dict):

	feature = np.zeros([len(user_dict), 10], dtype=np.float32)
	id_counter = 0
	est_date = datetime.fromisoformat('2006-03-21')
	for profile in user_dict.values():
		# 1) Verified?, 2) Enable geo-spatial positioning, 3) Followers count, 4) Friends count
		vector = [int(profile['verified']), int(profile['geo_enabled']), profile['followers_count'], profile['friends_count']]
		# 5) Status count, 6) Favorite count, 7) Number of lists
		vector += [profile['statuses_count'], profile['favourites_count'], profile['listed_count']]

		# 8) Created time (No. of months since Twitter established)
		user_date = datetime.strptime(profile['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
		month_diff = (user_date.year - est_date.year) * 12 + user_date.month - est_date.month
		vector += [month_diff]

		# 9) Number of words in the description, 10) Number of words in the screen name
		vector += [len(profile['name'].split()), len(profile['description'].split())]

		feature[id_counter, :] = np.reshape(vector, (1, 10))
		id_counter += 1
		print(id_counter)

	return feature