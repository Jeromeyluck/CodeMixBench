import os

shots = [0,1,2,5]
langs = ['onlyeng', 'chieng', 'spaeng', 'araeng', 'hineng', 'beneng', 'mareng', 'nepeng', 'freeng', 'tameng', 'duteng', 'gereng']
models = ['gpt-4o-ca', 'gpt-3.5-turbo-instruct', 'gpt-3.5-turbo', 'gpt-4-turbo-preview']
temperature = 0.8
api = 'sk-gwH6R3stxyGgajAeoqLD2zVIFzENsN1zuClfmYk4tEikW8ff'
url = 'https://api.chatanywhere.tech/v1'
#print(langs[:5])

os.system(f"python ./test_model.py --dataset lid_guaspa --expid lid_guaspa_all_0shot --model gpt-3.5-turbo --shot 0 --api {api} --url {url}")

