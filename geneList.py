from dataPreprocess import *
import os
#input
gene_expression_path = './Dataset/scRNA-Seq/mHSC-GM/ExpressionData.csv'
gene_order_path = './Dataset/scRNA-Seq/mHSC-GM/GeneOrdering.csv'
gold_network_path='./Dataset/Gold-network/Gold-network/STRING-network.csv'
#output
filtered_path= './Dataset/scRNA-Seq/mHSC-GM/'
FGN_file_name= 'mHSC-GM-STRING-FGN.csv'
rank_path= './DGRNS-main/Dataset/scRNA-Seq/mHSC-GM/'
Rank_file_name='mHSC-GM-200genes-STRING-rank.csv'
genePairList_path= './Dataset/scRNA-Seq/mHSC-GM/'
GPL_file_name= 'mHSC-GM-STRING-GPL.csv'

Rank_num=500

origin_expression_record,cells=get_origin_expression_data(gene_expression_path)
Expression_gene_num=len(origin_expression_record)
Expression_cell_num=len(cells)

low_express_gene_list=get_low_express_gene(origin_expression_record,len(cells))
print(str(len(low_express_gene_list))+' genes in low expression.')
for gene in low_express_gene_list:
    origin_expression_record.pop(gene)


if not os.path.isdir(rank_path):
    os.makedirs(rank_path)
rank_list,significant_gene_list = get_gene_ranking(gene_order_path, low_express_gene_list,Rank_num, rank_path + Rank_file_name, False)

gold_pair_record, gold_score_record ,unique_gene_list= get_filtered_gold(gold_network_path, rank_list,rank_path+Rank_file_name,True)

if not os.path.isdir(filtered_path):
    os.makedirs(filtered_path)
generate_filtered_gold( gold_pair_record, gold_score_record,filtered_path + FGN_file_name)


if not os.path.isdir(rank_path):
    os.makedirs(rank_path)
get_gene_pair_list(unique_gene_list, gold_pair_record, gold_score_record, genePairList_path + GPL_file_name)



