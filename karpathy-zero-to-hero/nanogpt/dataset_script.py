import requests

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

try:
    data = requests.get(url).text
    with open("input.txt" , "w") as f:
        f.write(data)
except Exception as e:
    print(f"Error: {e}")