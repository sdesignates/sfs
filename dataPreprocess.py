import pandas as pd
import numpy as np
import csv
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt

def get_tf_list(tf_path):
    # return tf_list
    f_tf = open(tf_path)
    tf_reader = list(csv.reader(f_tf))
    tf_list=[]
    for single in tf_reader[1:]:
        tf_list.append(single[0])
    print('Load '+str(len(tf_list))+' TFs successfully!')
    return tf_list


def get_origin_expression_data(gene_expression_path):
    # return 1.tf-targets dict and pair-score dict
    #        2.number of timepoints
    f_expression = open(gene_expression_path,encoding='utf-8')
    expression_reader = list(csv.reader(f_expression))
    cells = expression_reader[0][1:]
    num_cells = len(cells)

    expression_record = {}
    num_genes = 0
    for single_expression_reader in expression_reader[1:]:
        if single_expression_reader[0] in expression_record:
            exit('Gene name '+single_expression_reader[0]+' repeat!')
        expression_record[single_expression_reader[0]] = list(map(float, single_expression_reader[1:]))
        num_genes += 1
    print(str(num_genes) + ' genes and ' + str(num_cells) + ' cells are included in origin expression data.')
    return expression_record,cells


def get_normalized_expression_data(gene_expression_path):
    # return 1.tf-targets dict and pair-score dict
    #        2.number of timepoints
    expression_record,cells=get_origin_expression_data(gene_expression_path)
    expression_matrix = np.zeros((len(expression_record), len(cells)))
    index_row=0
    for gene in expression_record:
        expression_record[gene]=np.log10(np.array(expression_record[gene])+10**-2)
        expression_matrix[index_row]=expression_record[gene]
        index_row+=1

    #Heat map
    # plt.figure(figsize=(15,15))
    # sns.heatmap(expression_matrix[0:100,0:100])
    # plt.show()

    return expression_record, cells


def get_gene_ranking(gene_order_path,low_express_gene_list,gene_num,output_path,flag):#flag=True:write to output_path
    #1.delete genes p-value>=0.01
    #2.delete genes with low expression
    #3.rank genes in descending order of variance
    #4.return gene names list of top genes and variance_record of p-value<0.01
    f_order = open(gene_order_path)
    order_reader = list(csv.reader(f_order))
    if flag:
        f_rank = open(output_path, 'w', newline='\n')
        f_rank_writer = csv.writer(f_rank)
    variance_record = {}
    variance_list = []
    significant_gene_list=[]
    for single_order_reader in order_reader[1:]:
        # column 0:gene name
        # column 1:p value
        # column 2:variance
        if float(single_order_reader[1]) >= 0.01:
            break
        if single_order_reader[0] in low_express_gene_list:
            continue
        variance = float(single_order_reader[2])
        if variance not in variance_record:# 1 variance corresponding to 1 gene
            variance_record[variance] = single_order_reader[0]
        else:# 1 variance corresponding to n genes
            print(str(variance_record[variance]) + ' and ' + single_order_reader[0] + ' variance repeat!')
            variance_record[variance]=[variance_record[variance]]
            variance_record[variance].append(single_order_reader[0])
        variance_list.append(variance)
        significant_gene_list.append(single_order_reader[0])
    print('After delete genes with p-value>=0.01 or low expression, '+str(len(variance_list))+' genes left.')
    variance_list.sort(reverse=True)
    gene_rank = []
    for single_variance_list in variance_list[0:gene_num]:
        if type(variance_record[single_variance_list]) is str:# 1 variance corresponding to 1 gene
            gene_rank.append(variance_record[single_variance_list])
        else:# 1 variance corresponding to n genes
            gene_rank.append(variance_record[single_variance_list][0])
            del variance_record[single_variance_list][0]
            if len(variance_record[single_variance_list])==1:
                variance_record[single_variance_list]=variance_record[single_variance_list][0]
        if flag:
            f_rank_writer.writerow([variance_record[single_variance_list]])
    f_order.close()
    if flag:
        f_rank.close()
    return gene_rank,significant_gene_list


def get_filtered_gold(gold_network_path,rank_list,output_path,flag):
    #1.Load origin gold file
    #2.Delete genes not in rank_list
    #3.return tf-targets dict and pair-score dict
    #Note: If no score in gold network, score=999
    f_gold = open(gold_network_path,encoding='utf-8')
    gold_reader = list(csv.reader(f_gold))
    has_score=True
    if len(gold_reader[0])<3:
        has_score = False
    gold_pair_record = {}
    gold_score_record = {}
    unique_gene_list=[]
    for single_gold_reader in gold_reader[1:]:
        # column 0: TF
        # column 1: target gene
        # column 2: regulate score
        if (single_gold_reader[0] not in rank_list) or (single_gold_reader[1] not in rank_list):
            continue
        gene_pair = [single_gold_reader[0], single_gold_reader[1]]
        str_gene_pair = single_gold_reader[0] + ',' + single_gold_reader[1]
        if single_gold_reader[0] not in unique_gene_list: unique_gene_list.append(single_gold_reader[0])
        if single_gold_reader[1] not in unique_gene_list: unique_gene_list.append(single_gold_reader[1])
        if str_gene_pair in gold_score_record:
            exit('Gold pair repeat!')
        if has_score:
            gold_score_record[str_gene_pair] = float(single_gold_reader[2])
        else:
            gold_score_record[str_gene_pair] = 999
        if gene_pair[0] not in gold_pair_record:
            gold_pair_record[gene_pair[0]] = [gene_pair[1]]
        else:
            gold_pair_record[gene_pair[0]].append(gene_pair[1])
    #Some statistics of gold_network
    print(str(len(gold_pair_record)) + ' TFs and ' + str(
            len(gold_score_record)) + ' edges in gold_network consisted of genes in rank_list.')
    print(str(len(unique_gene_list))+' genes are common in rank_list and gold_network.')
    rank_density = len(gold_score_record) / (len(gold_pair_record) * (len(rank_list)))
    gold_density = len(gold_score_record) / (len(gold_pair_record) * (len(unique_gene_list)))
    print('Rank genes density = edges/(TFs*(len(rank_gene)-1))='+str(rank_density))
    print('Gold genes density = edges/(TFs*len(unique_gene_list))=' + str(gold_density))

    #write to file
    if flag:
        f_unique = open(output_path, 'w', newline='\n')
        f_unique_writer = csv.writer(f_unique)
        out_unique=np.array(unique_gene_list).reshape(len(unique_gene_list),1)
        f_unique_writer.writerows(out_unique)
        f_unique.close()
    return gold_pair_record,gold_score_record,unique_gene_list


def generate_filtered_gold(gold_pair_record,gold_score_record,output_path):
    # write filtered_gold to output_path
    f_filtered = open(output_path, 'w', newline='\n')
    f_filtered_writer = csv.writer(f_filtered)
    f_filtered_writer.writerow(['TF', 'Target', 'Score'])

    for tf in gold_pair_record:
        once_output = []
        for target in gold_pair_record[tf]:
            single_output = [tf, target, gold_score_record[tf + ',' + target]]
            once_output.append(single_output)
        f_filtered_writer.writerows(once_output)
    f_filtered.close()


def get_gene_pair_list(unique_gene_list, gold_pair_record, gold_score_record, output_file):
    # positive is relationship that tf regulate target
    # negtive is reationship that same tf doesn's regulate target.
    # When same tf doesn't have enough negtive, borrow negtive from other TFs.
    # When negtive is not enough,stop and prove positive:negtive = 1:1

    # generate all negtive gene pairs of TFs
    all_tf_negtive_record = {}
    for tf in gold_pair_record:
        all_tf_negtive_record[tf] = []
        for target in unique_gene_list:
            if target in gold_pair_record[tf]:
                continue
            all_tf_negtive_record[tf].append(target)

    # generate negtive record without borrow
    rank_negtive_record = {}
    for tf in gold_pair_record:
        num_positive = len(gold_pair_record[tf])
        if num_positive > len(all_tf_negtive_record[tf]):
            rank_negtive_record[tf] = all_tf_negtive_record[tf]
            all_tf_negtive_record[tf] = []
        else:
            #maybe random.sample(all_tf_negtive_record[tf],num_positive) to promote performance
            rank_negtive_record[tf] = all_tf_negtive_record[tf][:num_positive]
            all_tf_negtive_record[tf] = all_tf_negtive_record[tf][num_positive:]

    # output positive and negtive pairs
    f_gpl = open(output_file, 'w', newline='\n')
    f_gpl_writer = csv.writer(f_gpl)
    f_gpl_writer.writerow(['TF', 'Target', 'Label', 'Score'])
    stop_flag=False
    for tf in gold_pair_record:
        once_output = []
        for target in gold_pair_record[tf]:
            # output positive
            single_output = [tf, target, '1', gold_score_record[tf + ',' + target]]
            once_output.append(single_output)
            # output negtive
            if len(rank_negtive_record[tf]) == 0:
                # borrow negtive for other TFs
                find_negtive = False
                for borrow_tf in all_tf_negtive_record:
                    if len(all_tf_negtive_record[borrow_tf]) > 0:
                        find_negtive=True
                        single_output = [borrow_tf, all_tf_negtive_record[borrow_tf][0], 0, 0]
                        del all_tf_negtive_record[borrow_tf][0]
                        break
                # if not enough negtive of others,stop and prove positive:negtive = 1:1
                if not find_negtive:
                    stop_flag = True
                    break
            else:
                #negtive without borrow
                single_output = [tf, rank_negtive_record[tf][0], 0, 0]
                del rank_negtive_record[tf][0]
            once_output.append(single_output)
        if stop_flag:
            f_gpl_writer.writerows(once_output[:-1])
            print('Negtive not enough!')
            break
        f_gpl_writer.writerows(once_output)  # output positive and negtive of 1 TF at a time
    f_gpl.close()


def get_low_express_gene(origin_expression_record,num_cells):
    #get gene_list who were expressed in fewer than 10% of the cells

    gene_list=[]
    threshold=num_cells//10
    for gene in origin_expression_record:
        num=0
        for expression in origin_expression_record[gene]:
            if expression !=0:
                num+=1
                if num>threshold:
                    break
        if num<=threshold:
            gene_list.append(gene)
    return gene_list


