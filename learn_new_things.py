import json

filename = "intents.json"


def write_json(data, filename):
    with open(filename, "w", encoding="UTF-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def learn(tag=None, patterns=None, responses=None):
    with open(filename, encoding="UTF-8") as json_file:
        data = json.load(json_file)
        temp = data["intents"]
        y = {"tag": "{}".format(tag), "patterns": [f"{patterns}"], "responses": [f"{responses}"]}
        temp.append(y)
        write_json(data, filename)


def erweitern(tag=None, patterns=None, responses=None):
    with open(filename, encoding="UTF-8") as json_file:
        search = False
        intents = json.load(json_file)
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print("Oberbegriff " + tag + " gefunden.")
                search = True
                temp_p = intent["patterns"]
                temp_r = intent["responses"]
                temp_p.append(patterns)
                temp_r.append(responses)
                write_json(intents, filename)
                break

        if not search:
            print("Oberbegriff " + "\'" + tag + "\'" + " nicht gefunden.")
