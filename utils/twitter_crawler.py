import tweepy
import json

# Twitter Developer API tokens
auth = tweepy.OAuthHandler('xxx', 'xxx')
auth.set_access_token('xxx', 'xxx')

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

m, n = 0, 0
for i, user in enumerate(id_mappings):  # user id to twitter id mappings {user_id: twitter_account_id}
	try:
		# get recent 200 tweets of the user
		statuses = api.user_timeline(user_id=user, count=200)
		json_object = [json.dumps(s._json) + '\n' for s in statuses]
		# write the recent 200 tweet objects into a json file
		with open(str(user) + ".json", "w") as outfile:
			outfile.writelines(json_object)
		outfile.close()
	except tweepy.TweepError as err: # handle deleted/suspended accounts
		if str(err) == 'Not authorized.':  
			m+=1
			print(f'Not authorized: {m}')
		else:
			n+=1
			print(f'Page does not exist: {n}')
	print(f'user number: {i}')

print(f'Not authorized: {m}, Page does not exist: {n}.')