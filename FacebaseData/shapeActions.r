library(dplyr)
library(Hmisc)
library(stringr)
library(jsonlite)

setwd("/Users/loganyg/Documents/Summer_2017/CI/DatasetRecommender/FacebaseData")

# Load a mapping from filenames (FB...) to dataset IDs.
ds_id_map <- read.csv("facebase-events/accession.csv", header = FALSE, col.names = c("dataset_id", "filename"), colClasses = c("character", "character")) %>%
  mutate(filename = sub(" ", "", filename))
# Add some missing mappings.
ds_id_map <- ds_id_map %>%
  rbind(c(14068, "FB00000806")) %>%
  rbind(c(14130, "FB00000807"))

# Load view data from logs.
chaiserecord <- read.csv("facebase-events/chaiserecordevents.csv")
# Filter to only dataset accesses.
chaiserecord <- chaiserecord %>% filter(grepl(':dataset', ermrest_table), !grepl(':dataset_', ermrest_table))
# Create a mapping of events to the IDs of the dataset being accessed.
accessions_cr <- chaiserecord %>% filter(grepl('FB[0-9]+([.][0-9]+)?', filter)) %>%
  mutate(filter = paste("id=", ds_id_map$dataset_id[match(str_extract(filter, 'FB[0-9]+([.][0-9]+)?'), ds_id_map$filename)], sep=''))
# Combine the event->ID mapping with the events data.
chaiserecord <- rbind(filter(chaiserecord, !grepl('FB[0-9]+([.][0-9]+)?', filter)), accessions_cr)
chaiserecord <- chaiserecord %>% 
  mutate(dataset_id = str_extract(filter, '[0-9]+')) %>%
  mutate(datasource = paste(ermrest_table, filter, sep='_'))

# Load the download data from logs.
hatrac <- read.csv("facebase-events/hatracevents.csv")
# Filter to only dataset download requests that did not fail.
hatrac <- filter(hatrac, grepl('/hatrac/facebase/data/fb(1|2)/FB[0-9]+(.[0-9]+)?', path)) %>%
  filter(method == 'GET' | method == 'HEAD') %>%
  filter(status != 404)
# Add a column for the filename.
hatrac <- hatrac %>% 
  mutate(ds = str_extract(path, 'FB[0-9]+([.][0-9]+)?')) %>%
  rename(datasource = ds)

# Filter out IDs that don't have a mapping and add the action code 1 for a view.
chaiserecord <- chaiserecord %>% 
  filter(!is.na(match(chaiserecord$dataset_id, ds_id_map$dataset_id))) %>%
  mutate(action_type = 1)

# Filter out IDs that dont have a mapping and add the action code 4 for a download.
hatrac <- hatrac %>% 
  mutate(dataset_id = ds_id_map$dataset_id[match(datasource, ds_id_map$filename)]) %>%
  filter(!is.na(dataset_id)) %>%
  mutate(action_type = 4)

# Combine the two sources of data.
agg_data <- rbind(select(chaiserecord, id, client, user, dataset_id, action_type, devicereportedtime),
                  select(hatrac, id, client, user, dataset_id, action_type, devicereportedtime))
agg_data <- unique(agg_data) %>%
  mutate(user = as.character(user), client = as.character(client))

agg_data <- unique(agg_data)

# Create a DF for events with user associations.
user_agg_data <- agg_data %>%
  filter(user != "")

# Parse out the JSON that users are stored in.
res <- stream_in(textConnection(user_agg_data$user)) %>%
  mutate(globus_id = str_replace_all(id, 'https://auth.globus.org/', ''))

# Add the extracted user characteristics to the event data.
user_agg_data <- user_agg_data %>%
  mutate(email = res$email, full_name = res$full_name, globus_id = res$id) %>%
  mutate(globus_id = str_replace_all(globus_id, 'https://auth.globus.org/', ''))  

# A list of internal users that should not contribute to the model.
internal_users = c(
  "b7625859-1afd-444b-844d-805baef4ccfc",
  "b506963e-d274-11e5-99f0-67ee73dd4c3f",
  "abafeea0-d274-11e5-b053-4fd04d061aa4",
  "918e0956-d749-11e5-80bd-673bb1adf3db"
)

# Filter out internal users.
user_agg_data <- user_agg_data %>%
  filter(!(globus_id %in% internal_users))

# Write the full data to a csv with time data.
write.csv(
  rename(
    select(
      agg_data,
      client,
      dataset_id,
      action_type,
      devicereportedtime),
    source=client,
    target=dataset_id,
    weight=action_type,
    time=devicereportedtime
  ),
  'client_edgelist_formatted.csv')
# Write the full data to a csv.
write.csv(
  select(
    agg_data,
    client,
    dataset_id,
    action_type),
  'client_edgelist.csv')
# Write the user data to a csv with time data.
write.csv(
  rename(
    select(
      user_agg_data,
      globus_id,
      dataset_id,
      action_type,
      devicereportedtime),
    source=globus_id,
    target=dataset_id,
    weight=action_type,
    time=devicereportedtime
  ),
  'user_edgelist_formatted.csv')
# Write the user data to a csv.
write.csv(
  select(
    user_agg_data,
    globus_id,
    dataset_id,
    action_type),
  'user_edgelist.csv')
# This is the end of the parsing of log data to edgelists.
# The remaining code is for describing the data.
user_transfers <- agg_data %>%
  filter(user != "") %>%
  select(user, client, dataset_id, action_type) %>%
  unique() %>%
  mutate(user = as.character(user))
res <- stream_in(textConnection(user_transfers$user)) %>%
  mutate(globus_id = str_replace_all(id, 'https://auth.globus.org/', ''))
globus_map <- unique(res)
user_transfers <- user_transfers %>%
  mutate(email = res$email, full_name = res$full_name, globus_id = res$id) %>%
  mutate(globus_id = str_replace_all(globus_id, 'https://auth.globus.org/', ''))

user_actions <- user_transfers %>%
  select(email, globus_id, full_name, dataset_id, action_type) %>%
  unique()
user_views <- user_transfers %>%
  select(email, globus_id, full_name, dataset_id, action_type) %>%
  unique() %>%
  filter(action_type == 'view')
user_dls <- user_transfers %>%
  select(email, globus_id, full_name, dataset_id, action_type) %>%
  unique() %>%
  filter(action_type == 'download')

client_actions <- agg_data %>%
  select(client, dataset_id, action_type) %>%
  unique()
client_views <- agg_data %>%
  select(client, dataset_id, action_type) %>%
  unique() %>%
  filter(action_type == 'view')
client_dls <- agg_data %>%
  select(client, dataset_id, action_type) %>%
  unique() %>%
  filter(action_type == 'download')

user_action_freq <- data.frame(table(user_actions$globus_id)) %>%
  filter(Freq != 0, Var1 != "") %>%
  rename(datasets = Freq, globus_id = Var1) %>%
  mutate(email = globus_map$email[match(globus_id, globus_map$globus_id)]) %>%
  mutate(full_name = globus_map$full_name[match(globus_id, globus_map$globus_id)])
user_view_freq <- data.frame(table(user_views$globus_id)) %>%
  filter(Freq != 0, Var1 != "") %>%
  rename(datasets = Freq, globus_id = Var1) %>%
  mutate(email = globus_map$email[match(globus_id, globus_map$globus_id)]) %>%
  mutate(full_name = globus_map$full_name[match(globus_id, globus_map$globus_id)])
user_dl_freq <- data.frame(table(user_dls$globus_id)) %>%
  filter(Freq != 0, Var1 != "") %>%
  rename(datasets = Freq, globus_id = Var1) %>%
  mutate(email = globus_map$email[match(globus_id, globus_map$globus_id)]) %>%
  mutate(full_name = globus_map$full_name[match(globus_id, globus_map$globus_id)])
client_action_freq <- data.frame(table(client_actions$client)) %>%
  filter(Freq != 0) %>%
  rename(datasets = Freq, client = Var1)
client_dl_freq <- data.frame(table(client_dls$client)) %>%
  filter(Freq != 0) %>%
  rename(datasets = Freq, client = Var1)
client_view_freq <- data.frame(table(client_views$client)) %>%
  filter(Freq != 0) %>%
  rename(datasets = Freq, client = Var1)

latex(describe(select(user_action_freq, datasets), descript="Datasets Viewed or Downloaded per User"), file="./describeFacebase.tex")
latex(describe(select(user_dl_freq, datasets), descript="Datasets Downloaded per User"), file="./describeFacebase.tex", append=TRUE)
latex(describe(select(user_view_freq, datasets), descript="Datasets Viewed per User"), file="./describeFacebase.tex", append=TRUE)

latex(describe(select(client_action_freq, datasets), descript="Datasets Viewed or Downloaded per Client"), file="./describeFacebase.tex", append=TRUE)
latex(describe(select(client_dl_freq, datasets), descript="Datasets Downloaded per Client"), file="./describeFacebase.tex", append=TRUE)
latex(describe(select(client_view_freq, datasets), descript="Datasets Viewed per Client"), file="./describeFacebase.tex", append=TRUE)

user_clients <- agg_data %>%
  filter(user != "") %>%
  select(user, client) %>%
  unique()
user_client_freq <- data.frame(table(user_clients$user)) %>%
  filter(Freq != 0, Var1 != "") %>%
  rename(clients = Freq, user = Var1)
client_user_freq <- data.frame(table(user_clients$client)) %>%
  filter(Freq != 0, Var1 != "") %>%
  rename(users = Freq, client = Var1)

latex(describe(select(user_client_freq, clients), descript="Client per User"), file="./describeFacebase.tex", append=TRUE)
latex(describe(select(client_user_freq, users), descript="User per Client"), file="./describeFacebase.tex", append=TRUE)


