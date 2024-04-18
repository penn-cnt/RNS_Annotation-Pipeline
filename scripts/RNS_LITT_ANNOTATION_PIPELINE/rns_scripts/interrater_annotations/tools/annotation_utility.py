import pandas as pd

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import numpy as np
import matplotlib
from sklearn.metrics import accuracy_score

matplotlib.use("nbAgg")
from sklearn.decomposition import PCA
import data_utility
import interactive_plot
import times
import nltk

path = 'data'


def read_annotation(annotation_path='data/test_annotations.csv', annotation_catalog_path='data/test_datasets.csv',
                    data=None, n_class=2):
    annotations = pd.read_csv(annotation_path)
    data_catalog = pd.read_csv(annotation_catalog_path)
    return Annotations(annotations, data_catalog,data, n_class)


class Annotations:
    def __init__(self, raw_annotations, annotation_catalog, data, n_class):
        self.class_label = {}
        if n_class == 3:
            self.class_label = {"no": 0,
                                "yes": 1,
                                "maybe": 2}
        elif n_class == 2:
            self.class_label = {"no": 0,
                                "yes": 1,
                                "maybe": 1}
        self.raw_annotations = raw_annotations
        self.annotation_catalog = annotation_catalog
        self.HUP_IDs = list(self.raw_annotations['HUP_ID'].unique())
        self.data = data
        self.annotation_dict = {}

        self.raters = list(self.raw_annotations['dataset'].unique())

        self.type_annot = {}
        self.time_annot = {}
        self.process_annotations()
        self.raters = list(self.annotation_dict.keys())
        self.annotations = pd.concat([self.annotation_dict[key] for key in self.raters])

    def process_annotations(self):
        self.raw_annotations.rename(columns={self.raw_annotations.columns[0]: "annotID"}, inplace=True)
        self.convert_time()

        for rater in self.raters:
            rater_annotation = self.raw_annotations[self.raw_annotations['dataset'].str.contains(rater)]
            self.type_annot[rater] = rater_annotation[rater_annotation['layerName'].str.contains("isSeizure?")]
            self.time_annot[rater] = rater_annotation[rater_annotation['layerName'].str.contains("SeizureTimes")]
            self.clean_type_description(rater)
            self.clean_channel_description(rater)
            self.make_cleaned_annotation(rater)

    def convert_time(self):
        start_time = []
        end_time = []
        annot_time = self.raw_annotations['annots']
        for t in annot_time:
            test = t
            try:
                ind_comma = t.index(',')
            except:
                print()
            start_time.append(int(t[1:ind_comma]))
            end_time.append(int(t[ind_comma + 2:-1]))
        assert len(start_time) == len(annot_time)
        self.raw_annotations['Start Time'] = start_time
        self.raw_annotations['End Time'] = end_time

    def clean_type_description(self, rater):
        annot_class = np.empty(len(self.type_annot[rater]))
        type_description = list(self.type_annot[rater]['descriptions'])
        for index, type_string in enumerate(type_description):
            tokens = nltk.tokenize.word_tokenize(str(type_string).lower())
            token_dict_keys = list(self.class_label.keys())
            elements = []
            for tk in tokens:
                if tk in token_dict_keys:
                    elements.append(self.class_label[tk])
            if len(elements) == 1:
                annot_class[index] = elements[0]
            else:
                annot_class[index] = 3
        temp_df = self.type_annot[rater].copy()
        temp_df['label'] = annot_class.astype(int)
        self.type_annot[rater] = temp_df

    def clean_channel_description(self, rater):
        catalog_index_list = []
        channel_code_arr = np.zeros((len(self.time_annot[rater]), 4))
        binary_channel_code_arr = np.zeros((len(self.time_annot[rater]), 4))

        for index in range(len(self.time_annot[rater])):
            row = self.time_annot[rater].iloc[index]
            channel_description = row['descriptions']
            start_time = row['Start Time']
            end_time = row['End Time']
            patientID = row['HUP_ID']
            episode_start_time_series = \
                self.type_annot[rater][self.type_annot[rater]['HUP_ID'].str.contains(patientID)][
                    'Start Time']
            episode_end_time_series = self.type_annot[rater][self.type_annot[rater]['HUP_ID'].str.contains(patientID)][
                'End Time']
            catalog_ind_start = times.find_closest_event_ind(start_time, list(episode_start_time_series),
                                                             from_side='right')
            catalog_ind_end = times.find_closest_event_ind(end_time, list(episode_end_time_series),
                                                           from_side='left')
            try:
                assert catalog_ind_start == catalog_ind_end
                catalog_ind = episode_start_time_series.index[catalog_ind_start]
            except:
                original_end_time = end_time
                end_time_dummy = episode_end_time_series.iloc[catalog_ind_start]
                diff_end = original_end_time - end_time_dummy
                original_start_time = start_time
                start_time_dummy = episode_start_time_series.iloc[catalog_ind_end]
                diff_start = original_start_time - start_time_dummy

                if abs(diff_start) > abs(diff_end):
                    end_time = episode_end_time_series.iloc[catalog_ind_start]
                    temp_df = self.time_annot[rater].copy()
                    temp_df.loc[temp_df.index[index], 'End Time'] = end_time
                    self.time_annot[rater] = temp_df
                    catalog_ind = episode_start_time_series.index[catalog_ind_start]
                else:  # TODO check start end series
                    start_time = episode_start_time_series.iloc[catalog_ind_end]
                    temp_df = self.time_annot[rater].copy()
                    temp_df.loc[temp_df.index[index], 'Start Time'] = start_time
                    self.time_annot[rater] = temp_df
                    catalog_ind = episode_end_time_series.index[catalog_ind_end]

            row = self.time_annot[rater].iloc[index]
            start_time = row['Start Time']
            end_time = row['End Time']
            episode_start_time = self.raw_annotations.loc[catalog_ind, 'Start Time']
            episode_end_time = self.raw_annotations.loc[catalog_ind, 'End Time']
            try:
                assert start_time - episode_start_time >= 0 and episode_end_time - end_time >= 0
            except:
                print("error" + str(catalog_ind))

            catalog_index_list.append(catalog_ind)

            class_code = self.type_annot[rater].loc[catalog_ind, 'label']
            if any(s.isnumeric() for s in channel_description):
                for s in channel_description:
                    if s.isnumeric():
                        channel_code_arr[index, int(s) - 1] = class_code
                        binary_channel_code_arr[index, int(s) - 1] = 1
            else:
                channel_code_arr[index] = [3, 3, 3, 3]

        c = channel_code_arr.astype(int).tolist()
        cb = binary_channel_code_arr.astype(int).tolist()
        channel_code = [''.join(str(e) for e in line) for line in c]
        binary_channel_code = [''.join(str(e) for e in line) for line in cb]
        temp_df = self.time_annot[rater].copy()
        temp_df['Catalog Index'] = catalog_index_list
        temp_df['Channel Code'] = channel_code
        temp_df['Binary Channel Code'] = binary_channel_code
        self.time_annot[rater] = temp_df

    def make_cleaned_annotation(self, rater):
        type_annot = self.type_annot[rater]
        time_annot = self.time_annot[rater]
        annot_df = pd.DataFrame()
        annot_df["Dataset"] = type_annot['dataset']
        annot_df["Annotation_Catalog_Index"] = type_annot.index
        annot_df["Patient_ID"] = type_annot['HUP_ID']
        annot_df["Alias_ID"] = type_annot['aliasID']
        annot_df["Episode_Start_Timestamp"] = type_annot['Start Time']
        annot_df["Episode_End_Timestamp"] = type_annot['End Time']
        annot_df["Episode_Start_UTC_Time"] = [times.timestamp_to_utctime(ts) for ts in list(type_annot['Start Time'])]
        annot_df["Episode_End_UTC_Time"] = [times.timestamp_to_utctime(ts) for ts in list(type_annot['End Time'])]
        annot_df["Episode_Index"] = list(range(len(annot_df)))
        annot_df["Episode_Start_Index"] = list(range(len(annot_df)))
        annot_df["Episode_End_Index"] = list(range(len(annot_df)))
        annot_df["Annotation_Start_Timestamp"] = [list() for _ in range(len(annot_df))]
        annot_df["Annotation_End_Timestamp"] = [list() for _ in range(len(annot_df))]
        annot_df["Annotation_Start_UTC_Time"] = [list() for _ in range(len(annot_df))]
        annot_df["Annotation_End_UTC_Time"] = [list() for _ in range(len(annot_df))]
        annot_df["Annotation_Start_Index"] = [list() for _ in range(len(annot_df))]
        annot_df["Annotation_End_Index"] = [list() for _ in range(len(annot_df))]
        annot_df["Type_Description"] = type_annot['descriptions']
        annot_df["Class_Code"] = type_annot['label']
        annot_df["Annotation_Channel"] = [list() for _ in range(len(annot_df))]
        annot_df["Channel_Code"] = [list() for _ in range(len(annot_df))]
        annot_df["Binary_Channel_Code"] = [list() for _ in range(len(annot_df))]

        for index in range(len(time_annot)):
            row = self.time_annot[rater].iloc[index]
            catalog_ind = row['Catalog Index']
            annot_df_index = annot_df.loc[annot_df['Annotation_Catalog_Index'] == catalog_ind].index
            # ==========================================================================
            annot_df.loc[annot_df_index, 'Annotation_Start_Timestamp'].values[0].append(row['Start Time'])
            annot_df.loc[annot_df_index, 'Annotation_End_Timestamp'].values[0].append(row['End Time'])
            # ==========================================================================
            annot_df.loc[annot_df_index, 'Annotation_Start_UTC_Time'].values[0].append(
                times.timestamp_to_utctime(row['Start Time']))
            annot_df.loc[annot_df_index, 'Annotation_End_UTC_Time'].values[0].append(
                times.timestamp_to_utctime(row['End Time']))
            annot_df.loc[annot_df_index, 'Annotation_Start_Index'].values[0].append(
                times.timestamp_to_ind(row['Start Time'],
                                       self.data[row["HUP_ID"]].catalog)[1])
            annot_df.loc[annot_df_index, 'Annotation_End_Index'].values[0].append(
                times.timestamp_to_ind(row['End Time'],
                                       self.data[row["HUP_ID"]].catalog)[1])
            # ==========================================================================
            annot_df.loc[annot_df_index, 'Annotation_Channel'].values[0].append(row['descriptions'])
            annot_df.loc[annot_df_index, 'Channel_Code'].values[0].append(row['Channel Code'])
            annot_df.loc[annot_df_index, 'Binary_Channel_Code'].values[0].append(row['Binary Channel Code'])

        for index in range(len(annot_df)):
            row = annot_df.iloc[index]
            catalog_ind = row['Annotation_Catalog_Index']
            annot_df_index = annot_df.loc[annot_df['Annotation_Catalog_Index'] == catalog_ind].index
            try:
                clip_ind_start, start_ind = times.timestamp_to_ind(row['Episode_Start_Timestamp'],
                                                               self.data[row["Patient_ID"]].catalog)
            except:
                print()
            clip_ind_end, end_ind = times.timestamp_to_ind(row['Episode_End_Timestamp'],
                                                           self.data[row["Patient_ID"]].catalog)

            annot_df.loc[annot_df_index, "Episode_Index"] = clip_ind_start
            annot_df.loc[annot_df_index, "Episode_Start_Index"] = start_ind
            annot_df.loc[annot_df_index, "Episode_End_Index"] = end_ind

        print(len(annot_df))

        annot_df = self.error_checking(annot_df)

        if len(annot_df) != 0:
            self.annotation_dict[rater] = annot_df

    def error_checking(self, annot_df):
        to_drop_list = []
        print(len(annot_df))
        for index in range(len(annot_df)):
            is_code_valid = False
            is_annot_valid = False
            row = annot_df.iloc[index]
            class_code = row['Class_Code']
            type_description = row['Type_Description']
            channel_description = row['Annotation_Channel']
            channel_code = row['Channel_Code']
            is_code_valid = class_code != 3
            if is_code_valid:
                if class_code == 0:
                    is_annot_valid = True
                if class_code == 1 or class_code == 2:
                    is_annot_valid = len(channel_description) != 0
                if any([c == '3' for code in channel_code for c in code]):
                    is_annot_valid = False

            if is_annot_valid == False or is_code_valid == False:
                to_drop_list.append(row["Annotation_Catalog_Index"])
        print(len(annot_df))

        for catalog_index in to_drop_list:
            annot_df = annot_df.drop(annot_df.loc[annot_df['Annotation_Catalog_Index'] == catalog_index].index)

        return annot_df

    def annot_match(self, match_annotation='type'):
        pred_dict = {}
        catalog_dict = {}
        for teacher_key in self.annotation_dict:
            teacher_annotation = self.annotation_dict[teacher_key]
            pred_dict[teacher_key] = {}
            catalog_dict[teacher_key] = {}
            for student_key in self.annotation_dict:
                pred_dict[teacher_key][student_key] = []
                catalog_dict[teacher_key][student_key] = []
                student_annotation = self.annotation_dict[student_key]
                for i in range(len(teacher_annotation)):
                    teacher_row = teacher_annotation.iloc[i]
                    eps_start_time = teacher_row["Episode_Start_Timestamp"]
                    eps_end_time = teacher_row["Episode_End_Timestamp"]
                    patientID = teacher_row["Patient_ID"]
                    student_row = student_annotation.loc[
                        (student_annotation['Episode_Start_Timestamp'] == eps_start_time) & (
                                student_annotation['Episode_End_Timestamp'] == eps_end_time) & (
                                student_annotation['Patient_ID'] == patientID)]
                    if match_annotation == 'type':
                        if len(student_row == 1):
                            pred_dict[teacher_key][student_key].append(
                                [int(teacher_row['Class_Code']), int(student_row['Class_Code'])])
                            catalog_dict[teacher_key][student_key].append(
                                [int(teacher_row['Annotation_Catalog_Index']),
                                 int(student_row['Annotation_Catalog_Index'])])
                    if match_annotation == 'channel':
                        if len(student_row == 1) and teacher_row['Binary_Channel_Code'] != []:
                            pred_dict[teacher_key][student_key].append(
                                [teacher_row['Binary_Channel_Code'], student_row['Binary_Channel_Code'].item()])
                            catalog_dict[teacher_key][student_key].append(
                                [int(teacher_row['Annotation_Catalog_Index']),
                                 int(student_row['Annotation_Catalog_Index'])])
                    if match_annotation == 'time':
                        if len(student_row == 1) and teacher_row['Binary_Channel_Code'] != []:
                            pred_dict[teacher_key][student_key].append(
                                [(teacher_row['Annotation_Start_Index'], teacher_row['Annotation_End_Index']), (
                                    student_row['Annotation_Start_Index'].item(),
                                    student_row['Annotation_End_Index'].item())])
                            catalog_dict[teacher_key][student_key].append(
                                [int(teacher_row['Annotation_Catalog_Index']),
                                 int(student_row['Annotation_Catalog_Index'])])

        return pred_dict, catalog_dict


def annot_type_accuracy(pred_dict):
    result_dict = {}
    result_arr = np.empty((len(pred_dict), len(pred_dict)))
    print(result_arr.shape)
    for i, teacher_key in enumerate(pred_dict):
        result_dict[teacher_key] = {}
        for j, student_key in enumerate(pred_dict[teacher_key]):
            pred_arr = np.array(pred_dict[teacher_key][student_key])
            acc = accuracy_score(pred_arr[:, 0], pred_arr[:, 1])
            result_dict[teacher_key][student_key] = acc
            result_arr[i, j] = acc

    return result_dict, result_arr


def annot_channel_accuracy(pred_dict):
    channel_result_dict = {}
    channel_result_arr = np.empty((len(pred_dict), len(pred_dict)), dtype=object)
    print(channel_result_arr.shape)
    for i, teacher_key in enumerate(pred_dict):
        channel_result_dict[teacher_key] = {}
        for j, student_key in enumerate(pred_dict[teacher_key]):
            pred_arr = pred_dict[teacher_key][student_key]
            acc = sum([1 for k in range(len(pred_arr)) if pred_arr[k][0] == pred_arr[k][1]]) / len(pred_arr)
            channel_result_dict[teacher_key][student_key] = acc
            channel_result_arr[i, j] = acc

    return channel_result_dict, channel_result_arr


def annot_time_overlap(pred_dict):
    result_dict = {}
    result_arr = np.empty((len(pred_dict), len(pred_dict)), dtype=object)
    for i, teacher_key in enumerate(pred_dict):
        result_dict[teacher_key] = {}
        for j, student_key in enumerate(pred_dict[teacher_key]):
            total_overlap_time = 0
            total_total_time = 0
            pred_arr = pred_dict[teacher_key][student_key]
            for pair in pred_arr:
                start_1_list = pair[0][0]
                end_1_list = pair[0][1]
                start_2_list = pair[1][0]
                end_2_list = pair[1][1]
                overlap_time = 0
                total_time = 0
                for l1 in range(len(start_1_list)):
                    for l2 in range(len(start_2_list)):
                        overlap_time += find_overlap(start_1_list[l1], end_1_list[l1], start_2_list[l2], end_2_list[l2])
                try:
                    total_time = max(max(end_1_list), max(end_2_list)) - min(min(start_1_list), min(start_2_list)) + 1
                except:
                    total_time = 0
                total_overlap_time += overlap_time
                total_total_time += total_time
            result_dict[teacher_key][student_key] = total_overlap_time / total_total_time
            result_arr[i, j] = total_overlap_time / total_total_time

    return result_dict, result_arr


def find_overlap(start_1, end_1, start_2, end_2):
    return max(0, min(end_1, end_2) - max(start_1, start_2) + 1)


def check_diff(predict_dict, catalog_dict, teacher_key="BrianLitt_RNS_Test_Dataset"):
    diff_cataglog_ind = []
    diff_predict = []
    for j, student_key in enumerate(predict_dict[teacher_key]):
        pred_arr = np.array(predict_dict[teacher_key][student_key])
        for k in range(len(pred_arr)):
            if pred_arr[k][0] != pred_arr[k][1]:
                diff_cataglog_ind.append(catalog_dict[teacher_key][student_key][k])
                diff_predict.append(predict_dict[teacher_key][student_key][k])
    return diff_predict, diff_cataglog_ind


def sort_overlap(predict_dict, catalog_dict, teacher_key="BrianLitt_RNS_Test_Dataset"):
    start_time_diff = []
    end_time_diff = []
    diff_catalog_ind = []
    for j, student_key in enumerate(predict_dict[teacher_key]):
        total_overlap_time = 0
        total_total_time = 0
        pred_arr = predict_dict[teacher_key][student_key]
        for k, pair in enumerate(pred_arr):
            try:
                start_1 = pair[0][0][0]
                end_1 = pair[0][1][0]
                start_2 = pair[1][0][0]
                end_2 = pair[1][1][0]
            except:
                continue
            start_time_diff.append(abs(start_1 - start_2))
            end_time_diff.append(abs(end_1 - end_2))
            diff_catalog_ind.append(catalog_dict[teacher_key][student_key][k])

    return start_time_diff, end_time_diff, diff_catalog_ind
