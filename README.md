PHASE 1: Project Initialised
worked on creating the project structure, and installed the requied lib.

Phase 2: DataFrame creation
raw data collection, data summary for each csv file created, macroeconomic features downloaded, Inspection modified

Phase 3: Build features using stockstats usingacquired raw data and push it on the prcoessed folder along aith forward filling and merging the macro features to the excisting asset dataframe
Also, decided to go with GSPC as a basis to create the Target values for the dataset, apply a LGBM to rank important features, accquired top features

Phase 4: Start creating the embeddings for the graph construction, and finialising the nodes and edges of the graph
Graph Constructed

Phase 5:GraphSage layer called, GraphSage Model created and Graph visualised

Phase 6: Form the ftructre of the PPO model, set the environment, the actions, rewards and next updates

Phase 7: PPO trained acc to graph embeddings

phase 8: adding realism to the PPO rewards

phase 9: updating the hyperparameters for the ppo model, caclulating financial metrics
