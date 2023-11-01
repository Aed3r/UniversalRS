##Towards a Universal Recommender System: A Linked Open Data Approach

#This repository contains the code used in the paper "Towards a Universal Recommender System: A Linked Open Data Approach" by G. Hubert

Abstract:
Recommender systems (RSs) play a crucial role in helping users make informed decisions in the face of an ever-increasing array of choices. Most RSs are domain-specific, but domain-independent (or general purpose) RSs can be applied to a wide range of application areas, leveraging data and insights from different domains and handling large user populations. Current RS research is dominated by "improve the state-of-the-art", in terms of accuracy, speed and scale. Truly domain-independent recommender systems currently represent a gap in research, as is shown in this work. 

This paper explores the development of a general-purpose RS and the use of LOD to integrate data from different domains in an unsupervised way. Building a system that can effectively generate recommendations for any domain without prior knowledge of the data is challenging. Linked Open Data (LOD) offers a solution to this problem by enabling the integration of data from multiple domains. We explore and evaluate various unsupervised methods for acquiring and sorting through data, and using it to generate accurate recommendations. The resulting product is a truly domain-independent recommendation framework that can, in theory, be applied to a large variety of use-cases without requiring modification. Finally, this paper suggests future research directions for building more effective general-purpose RSs.

## Installation and use of the GPRS framework

#Install required packages (using python 3.9 ideally):
```
pip install -r requirements.txt
```
Clone and install the latest version of pyrdf2vec here: https://github.com/IBCNServices/pyRDF2Vec/

#Dataset:
- Download ml-1m.zip from http://files.grouplens.org/datasets/movielens/ml-1m.zip and unzip to ./data/ml-1m/
- Download last.fm dataset from https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip and unzip to ./data/lastFM/
- Clone https://github.com/sisinflab/LODrecsys-datasets to ./data/LODrecsys-datasets/
- (optionally) create a DBpedia clone using blazegraph. Put the Blazegraph .jar and .jnl files in ./blazegraph or adjust the files to the correct location in the config file.

#Files:
- GPRS/sparql_endpoint.py: contains the SPARQL endpoint class. Connects to a SPARQL endpoint and can launch blazegraph too. Contains many utility functions such as type and attribute retrieval, candidates selection, etc. See the 'main' for examples.
- GPRS/recommendationExp.py: contains a full pipeline from loading data to evaluating the recommender system. See the 'main' for examples.
- GPRS/mappingExp.py: contains a full pipeline from loading data to evaluating the mapping system, with or without type filtering. See the 'main' for examples.
- GPRS/mappingExpBasic.py: simple mapping system.
- GPRS/jobPoster.py: contains utilities to sync files, dependencies, and launch and monitor jobs on the Gemini servers.
- GPRS/localJobPoster.py: contains utilities to launch and monitor jobs on grid engine running on the local machine.

#Results:
- All results go into ./data/results/
- The results used to in the paper can be found ./results

#Useful commands:
Account storage size:
`quota -s`

Use databus.jar to download dbpedia files:
`java -jar databus-client-v2.1-beta.jar -s query.sparql -e https://databus.dbpedia.org/sparql -t ./converted`

Open blazegraph interface via SSH tunnel:
`ssh -L 8080:10.20.51.34:9999 user@gemini.science.uu.nl` and go to http://localhost:8080/blazegraph/

Run Blazegraph:
`java -server -Xmx70g -Djetty.port=19999 -jar blazegraph.jar` 

Find what process is using blazegraph.jnl (kill to start new blazegraph instance):
`fuser -v blazegraph.jnl`

Find what process is using the port:
`lsof -i:19999`

Grid engine monitoring:
Details/usage of job: `qstat -j <job_id>`
Hosts status: `qhost`
Hosts status with details: `qhost -F`
Job summary after run: `qacct -j <job_id>`