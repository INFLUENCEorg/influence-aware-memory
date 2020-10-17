from pymongo import MongoClient
import json
import gridfs
import os
from sacred.dependencies import get_digest
import shutil
import argparse

# The MongoObserver output creates files for 4 different collections, "runs", "metrics", "fs.chunks", and "fs.files".
# runs: contains most of the information of the experiments, points to related "metrics", "fs.chunks", and "fs.files"
# metrics: for each metric that is tracked in an experiment a separate entry into the "metrics" collection is made
# fs.files: for each file that makes up the source code of the experiment an entry is made that contains a hash code,
#           the path of the file, and points to the related "fs.chunks" document
# fs.chunks: Contains a binary of the code

def createRunsDocument(db, experimentFolder):
    with open(experimentFolder + '/run.json') as data_file:
        runs_object = json.load(data_file)

    # if runs_object['status'] == "RUNNING":
        # return runs_object, True

    runs_object['format'] = "FS_to_Mongo"

    with open(experimentFolder + '/cout.txt') as data_file:
        runs_object['captured_out'] = data_file.read()

    with open(experimentFolder + '/config.json') as data_file:
        config = json.load(data_file)
    runs_object['config'] = config

    # Check which ID to give, then insert with id into database
    runs_object["_id"] = db.runs.count_documents({}) + 1
    db.runs.insert_one(runs_object)
    return runs_object, False

def createMetricsDocument(db, experimentFolder, run_id):
    with open(experimentFolder + '/metrics.json') as data_file:
        metrics = json.load(data_file)

    runs_metrics = []
    for key in metrics:
        metric = {
            'name': key,
            'run_id': run_id,
            'steps': metrics[key]['steps'],
            'timestamps': metrics[key]['timestamps'],
            'values': metrics[key]['values']
        }
        metric_id = db.metrics.insert_one(metric).inserted_id
        runs_metrics.append({"id": metric_id, "name": key})

    db.runs.find_one_and_update({"_id": run_id}, {"$set": {"info": {"metrics": runs_metrics}}})
    # breakpoint()

def createFsDocuments(db, runs_object, base_dir):
    fs = gridfs.GridFS(db)
    sources = []

    for source_name, source_path in runs_object['experiment']['sources']:
        fileInDb = False
        abs_path = os.path.join(base_dir, source_path)

        # Check if the file is already in the database
        if fs.exists(filename=abs_path):
            md5hash = get_digest(abs_path)
            # Check if it's the same version of the code
            if fs.exists(filename=abs_path, md5=md5hash):
                file = fs.find_one({'filename': abs_path, 'md5': md5hash})
                source = [source_name, file._id]
                if source not in sources:
                    sources.append(source)
                fileInDb = True
        if not fileInDb:
            with open(abs_path, 'rb') as file:
                file_id = fs.put(file, filename=abs_path)
            sources.append([source_name, file_id])

    db.runs.find_one_and_update({"_id": runs_object["_id"]}, {"$set": {"experiment.sources": sources}})

def parseArguments(parser):
    parser.add_argument("-i", "--input", help="Directory where Sacred FS output is stored",
                        default='./saved_runs')
    parser.add_argument("-c", "--client", help="Specifies the MongoClient, e.g. ",
                        default='mongodb://localhost:27017/')
    parser.add_argument("-db", "--database", help="Name of the database the results should be stored",
                        default="influence-aware-memory")

    return parser.parse_args()

def main():
    args = parseArguments(argparse.ArgumentParser())

    # Now it won't crash if you don't end your input path with a '/'
    if not args.input[-1] == '/':
        args.input += '/'

    client = MongoClient(args.client)
    db = client[args.database]

    experiments = os.listdir(args.input)
    sourcesExist = "_sources" in experiments
    if sourcesExist:
        experiments.remove("_sources")
    experiments.sort(key = lambda x: int(x))

    sourcesInUse = []

    for experiment in experiments:
        runs_object, running = createRunsDocument(db, args.input + experiment)
        print(running)
        # if not running:
        # breakpoint()
        createMetricsDocument(db, args.input + experiment, runs_object["_id"])
        createFsDocuments(db, runs_object, args.input)
        shutil.rmtree(args.input + experiment)
        # else:
            # for source_name, source_path in runs_object['experiment']['sources']:
                # sourcesInUse.append(source_path[9::]) # removes "_sources/" from the name

    if sourcesExist:
        sources = os.listdir(args.input + "_sources")
        for file in sources:
            if file not in sourcesInUse:
                os.remove(args.input+ "_sources/" + file)


if __name__ == "__main__":
    main()