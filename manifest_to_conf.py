import os
import argparse
import json
import yaml
import collections
import copy

def foldr_meta(s, m):
    for k, v in m.items():
        s = s.replace("{" + k + "}", str(v))
    return s

def update(d, u, meta=None):
    d = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v, meta=meta)
        elif (not meta is None) and isinstance(v, str):
            d[k] = foldr_meta(v, meta)
        else:
            d[k] = v
    return d

def all_combinations(items, template):
    '''
    Items IS-A list of lists of dictionaries
    template IS-A dictionary

    1. Compute a list of dictionaries.
    2. Each dictionary is the dictionary union of of sub-dicts making each possible selection of 1 dict from the lists
    '''
    combinations = [{}]
    for item in items:
        combinations = [update(copy.deepcopy(c), i) for c in combinations for i in item]
    return [update(template, c) for c in combinations]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('-m', type=str, help="Manifest file name")
    parser.add_argument('-t', type=str, help="Path to template")
    parser.add_argument('-o', type=str, help="Path to output")
    parser.add_argument('-n', type=str, help="Template for config names including placeholders of the form {tag} where tag is a key in meta[]")
    args = parser.parse_args()

    with open(args.m) as manifest_file:
        manifest = json.loads(manifest_file.read())

    with open(args.t) as template_file:
        template = yaml.load(template_file)

    confs = []
    for task in manifest["manifest"]:
        if(task["todo"] == "all_combinations"):
            C = all_combinations(task["items"], template)
        elif(task["todo"] == "each"):
            C = [update(template, t) for t in task["items"]]

        if("instance" in task.keys()):
            C = [update(c, task["instance"], meta=c["meta"]) for c in C]
        confs = confs + C

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    for c in confs:
        fname = foldr_meta(args.n, c["meta"])
        with open(os.path.join(args.o, fname), "w+") as cnf_file:
            cnf_file.write(yaml.dump(c))

