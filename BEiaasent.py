import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report
import re

user1 = pd.read_excel("20181107-newindianexpress_sentence_classification_user1_user3.xlsx")
user2 = pd.read_excel("20181107-newindianexpress_sentence_classification_user2_user4.xlsx")
user3 = pd.read_excel("20181107-newindianexpress_sentence_classification_user3_user1.xlsx")
user4 = pd.read_excel("20181107-newindianexpress_sentence_classification_user4_user2.xlsx")

def get_scores(y1,y2):

    print("Cohen's Kappa :")
    print(cohen_kappa_score(y1,y2))
    print("Classification Report :")
    print(classification_report(y1,y2))
    print("Krippendorff's Alpha :")
    print(calculate_Kalpha2(y1,y2))

def calculate_Kalpha2(y1,y2):
    df = pd.concat([y1,y2], axis=1)
    df.columns = ['y1', 'y2']
    agg = 0
    diss = 0
    for label in df.y1.unique().tolist():
        agg += 2*len(df[(df.y1 == label) & (df.y2 == label)])
        asd = len(df[df.y1 == label]) + len(df[df.y2 == label])
        diss += asd*(asd - 1)

    nom = (2*len(df)-1)*agg - diss
    denom = 2*len(df)*(2*len(df) - 1) - diss
    return nom/denom

def calculate_Kalpha1(y1,y2):

    observed_nom = 0
    expected_nom = 0
    expected_denom = 0
    y1 = y1.tolist()
    y2 = y2.tolist()

    print(y1)
    print(y2)
    for i,g in enumerate(y1):
        if g != y2[i]:
            observed_nom += 1

    observed_denom = len(y1)

    observed = float(observed_nom / observed_denom)

    y1.extend(y2)
    expected = 0.0
    for i,g in enumerate(y1):
        seenItself = False
        for j,h in enumerate(y1):
            if i == j and not seenItself:
                seenItself = True
                continue

            leng = 1
            lenh = 1
            if g == h:
                metric = 0
            else:
                metric = 1

            expected_nom += leng * leng + lenh * lenh + leng * lenh * metric
            expected_denom += leng + lenh

    if expected_denom != 0:
        expected = float(expected_nom / expected_denom)

    if expected == 0:
        #        return {"kalpha":0.0,"weight":observed_denom}
        return 0.0

    kalpha = 1.0 - observed / expected
    #    return {"kalpha":kalpha,"weight":observed_denom}
    return kalpha
    
user1 = user1[:993]
user3 = user3[:993]
user2 = user2[:1012]
user4 = user4[:1012]

print("user1 : " + str(len(user1)))
print("user3 : " + str(len(user3)))
print("user2 : " + str(len(user2)))
print("user4 : " + str(len(user4)))

user1 = user1[user1.label.isin([1,0,1.0,0.0,2,2.0])]
user2 = user2[user2.label.isin([1,0,1.0,0.0,2,2.0])]
user3 = user3[user3.label.isin([1,0,1.0,0.0,2,2.0])]
user4.columns =['url', 'sent_num', 'sentence', 'label', 'comment']
user4 = user4[user4.label.isin([1,0,1.0,0.0,2,2.0])]

#print(user4)
#user4.drop(user4.index[[11605]], inplace=True) # user3 didn't label 11605
#user4 = user4.loc[user3.index]

user4.label = user4.label.astype('int64')
user3.label = user3.label.astype('int64')
user1.label = user1.label.astype('int64')
user2.label = user2.label.astype('int64')

#print(user3[~user3.label.isin([1,0,1.0,0.0,2,2.0])])

print("AFTER : ")
print("user1 : " + str(len(user1)))
print("user3 : " + str(len(user3)))
print("user2 : " + str(len(user2)))
print("user4 : " + str(len(user4)))

print("Bağlan & user3")
get_scores(user1.label,user3.label)

print("user2 & user4")

'''
zeros = user4.isin([0,0.0]).any(axis=1).apply(lambda x: 0.0 if x else np.nan)
ones = user4.isin([1,1.0]).any(axis=1).apply(lambda x: 1.0 if x else np.nan)
twos = user4.isin([2,2.0]).any(axis=1).apply(lambda x: 2.0 if x else np.nan)
alla = zeros.combine(ones, lambda x1, x2: x1 if np.isnan(x2) else x2)
alla = alla.combine(twos, lambda x1, x2: x1 if np.isnan(x2) else x2)
alla = alla.drop([10050])
user3 = user3.drop([10050])
'''

get_scores(user2.label,user4.label)


user1.columns = ['url', 'sent_num', 'sentence', 'user1_label', 'user3_label']
user1['user1_comment'] = ''
user1['user3_comment'] = ''
user3.columns =['url', 'sent_num', 'sentence', 'label', 'comment']
user1.user1_comment = user1.user3_label
user1.user3_label = user3.label
user1.user3_comment = user3.comment
BE_diss = user1[user1.user1_label != user1.user3_label]

BE_agree = user1[user1.user1_label == user1.user3_label]
print("BE agree length" + str(len(BE_agree)))
print("BE 1's " + str(len(BE_agree[BE_agree.user1_label == 1])))
print("BE 0's " + str(len(BE_agree[BE_agree.user1_label == 0])))
print("BE 2's " + str(len(BE_agree[BE_agree.user1_label == 2])))


#create text
BE_diss.insert(loc=3, column='text', value='')
user3 = pd.read_excel("20181107-newindianexpress_sentence_classification_user3_user1.xlsx")
for index,row in BE_diss.iterrows():
    BE_diss.loc[index,'text'] = ' '.join(user3[user3.sent_num.str.contains('^' + re.sub(r"-\d+$", r"-", row.sent_num), regex=True)].sentence.tolist())

print(BE_diss)

writer = pd.ExcelWriter('20181107_user1_user3_sentence_adjudication.xlsx')
#BB_agree.to_excel(writer,"Agreements")
BE_diss.to_excel(writer,"Disagreements")
#BB_spotcheck.to_excel(writer,"Spotcheck")
#writer.save()



#Adjudication part
BE_agree.insert(loc=3, column='text', value='')
user3 = pd.read_excel("20181107-newindianexpress_sentence_classification_user3_user1.xlsx")
for index,row in BE_agree.iterrows():
    BE_agree.loc[index,'text'] = ' '.join(user3[user3.sent_num.str.contains('^' + re.sub(r"-\d+$", r"-", row.sent_num), regex=True)].sentence.tolist())

BE_spotcheck = BE_agree[BE_agree.user1_label == 1].sample(n=50)
BE_spotcheck = BE_spotcheck.append(BE_agree[BE_agree.user1_label == 0].sample(n=50))
BE_spotcheck = BE_spotcheck.append(BE_agree[BE_agree.user1_label == 2].sample(n=min(len(BE_agree[BE_agree.user1_label == 2]),25)))

writer = pd.ExcelWriter('20181107_user1_user3_sentence_adjudication.xlsx')
BE_agree.to_excel(writer,"Agreements")
BE_diss.to_excel(writer,"Disagreements")
BE_spotcheck.to_excel(writer,"Spotcheck")
writer.save()



user4.columns = ['url', 'sent_num', 'sentence', 'user4_label', 'user2_label']
user4['user4_comment'] = ''
user4['user2_comment'] = ''
user2.columns =['url', 'sent_num', 'sentence', 'label', 'comment']
user4.user4_comment = user4.user2_label
user4.user2_label = user2.label
user4.user2_comment = user2.comment
EB_diss = user4[user4.user4_label != user4.user2_label]

EB_agree = user4[user4.user4_label == user4.user2_label]
print("EB agree length" + str(len(EB_agree)))
print("EB 1's " + str(len(EB_agree[EB_agree.user4_label == 1])))
print("EB 0's " + str(len(EB_agree[EB_agree.user4_label == 0])))
print("EB 2's " + str(len(EB_agree[EB_agree.user4_label == 2])))


#create text
EB_diss.insert(loc=3, column='text', value='')
user2 = pd.read_excel("20181107-newindianexpress_sentence_classification_user2_user4.xlsx")
for index,row in EB_diss.iterrows():
    EB_diss.loc[index,'text'] = ' '.join(user4[user4.sent_num.str.contains('^' + re.sub(r"-\d+$", r"-", row.sent_num), regex=True)].sentence.tolist())

#print(EB_diss)


writer = pd.ExcelWriter('20181107_user4_user2_sentence_disagreements.xlsx')
EB_diss.to_excel(writer)
#writer.save()



#Adjudication part
EB_agree.insert(loc=3, column='text', value='')
user2 = pd.read_excel("20181107-newindianexpress_sentence_classification_user2_user4.xlsx")
for index,row in EB_agree.iterrows():
    EB_agree.loc[index,'text'] = ' '.join(user2[user2.sent_num.str.contains('^' + re.sub(r"-\d+$", r"-", row.sent_num), regex=True)].sentence.tolist())

EB_spotcheck = EB_agree[EB_agree.user4_label == 1].sample(n=50)
EB_spotcheck = EB_spotcheck.append(EB_agree[EB_agree.user4_label == 0].sample(n=50))
EB_spotcheck = EB_spotcheck.append(EB_agree[EB_agree.user4_label == 2].sample(n=min(len(BE_agree[BE_agree.user1_label == 2]),25)))

writer = pd.ExcelWriter('20181107_user4_user2_sentence_adjudication.xlsx')
EB_agree.to_excel(writer,"Agreements")
EB_diss.to_excel(writer,"Disagreements")
EB_spotcheck.to_excel(writer,"Spotcheck")
writer.save()


'''
user3.columns = ['url', 'sent_num', 'sentence', 'user3_label', 'user4_label']
user3.user4_label = user4.label
diss = user3[user3.user3_label != user3.user4_label]

#create text
diss.insert(loc=3, column='text', value='')
user4 = pd.read_excel("20181010-newindianexpress_sentence_classification_user4.xlsx")
for index,row in diss.iterrows():
    diss.loc[index,'text'] = ' '.join(user4[user4.sent_num.str.contains('^' + re.sub(r"-\d+$", r"-", row.sent_num), regex=True)].sentence.tolist())

writer = pd.ExcelWriter('20181010_user3_user4_sentence_disagreements.xlsx')
diss.to_excel(writer)
writer.save()
'''
