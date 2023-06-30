#mengimport library yang akan digunakan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori
import streamlit as st


st.title("Asosiasi Dengan Algoritma Apriori")

st.title("Print Data")
data = pd.read_csv('Groceries data.csv')
st.write(data)


data['Date'] = pd.to_datetime(data['Date'])
st.write(data)

#Cek Data
st.title("Cek Data")
data.isnull().sum()
data = data.drop_duplicates()
st.write(data)

#Mensortir Data Berdasarkan Nomer Member Dan Tanggal
st.title("Mensortir Data")
data_sorted = data.groupby(['Member_number','Date']).agg({'itemDescription': lambda x: ','.join(x)}).reset_index()
st.write(data_sorted)

#Membuat List Item Description
st.title("List Item Description")
transactions =[]
for row in range(0,len(data_sorted)):
    transactions.append(data_sorted['itemDescription'][row].split(','))
transactions[:2]
st.write(transactions)

#Membuat Aturan Algoritma Apriori
rules = apriori(transactions = transactions, min_support = 0.001, min_confidence = 0.1, min_lift = 1, min_length = 2, max_length = 2)
# Menggunakan aturan support > 0.001 ; confidence > 0.1 ; lift > 1 ; kombinasi dari 2 barang
results = list(rules)
st.write(tuple(results))

#Membuat Tabel Rules Algoritma Apriori
def inspect(results):
    brg1 = [tuple(result[2][0][0])[0] for result in results]
    brg2 = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(brg1, brg2, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns=['Barang ke - 1', 'Barang ke - 2', 'Support', 'Confidence', 'Lift'])
st.write(resultsinDataFrame)

#Melakukan Sorting Hasil Perhitungan Dari Nilai Lift Tertinggi
resultsinDataFrame_sort = resultsinDataFrame.sort_values(by = ['Lift'], ascending = False)
st.write(resultsinDataFrame_sort)