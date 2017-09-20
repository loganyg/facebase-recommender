# August 18, 2017
# Logan Young
#
# A quick script to convert an edgelist in csv format
# to a an adjacency matrix.

import csv
import pandas as pd


def edgelist_to_adjmatrix(source,
                          target,
                          src_col,
                          tar_col,
                          dl_score=3,
                          view_score=2,
                          dl_atype='download',
                          view_atype='view'):
    clientset = set()
    dsset = set()
    # First Pass
    with open(source, 'r') as edgelist:
        el_reader = csv.DictReader(edgelist)
        for line in el_reader:
            client = line[src_col]
            ds = line[tar_col]
            if client not in clientset:
                clientset.add(client)
            if ds not in dsset:
                dsset.add(ds)
        adj_matrix = pd.DataFrame(index=clientset, columns=dsset, data=0)
    # Second Pass
    with open(source, 'r') as edgelist:
        el_reader = csv.DictReader(edgelist)
        for line in el_reader:
            client = line[src_col]
            ds = line[tar_col]
            atype = line['action_type']
            if atype == view_atype:
                if adj_matrix[ds][client] in (0, dl_score):
                    adj_matrix[ds][client] += view_score
            elif atype == dl_atype:
                if adj_matrix[ds][client] in (0, view_score):
                    adj_matrix[ds][client] += dl_score
    adj_matrix.to_csv(target)


def adjmatrix_to_edgelist(source,
                          target):
    adjmatrix = pd.read_csv(source)
    with open(target, 'w') as out:
        writer = csv.writer(out)
        writer.writerow(['source', 'target', 'weight'])
        for i in adjmatrix.index:
            for j in adjmatrix.columns:
                writer.writerow([i, j, adjmatrix.ix[i, j]])
