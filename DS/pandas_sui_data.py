import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"C:\Users\jlueck\Box\JLueck\vsc\python\DS\sui_data.csv") 

df = df[df['STUB_NAME'] == 'Sex']
df = df[df['YEAR'].between(2000, 2018)]
df.drop(columns=['INDICATOR', 'UNIT', 'UNIT_NUM', 'STUB_NAME_NUM', 'STUB_LABEL_NUM', 'AGE_NUM', 'YEAR_NUM', 'FLAG', 'STUB_NAME', 'AGE'], inplace=True)

male_df = df[df['STUB_LABEL'] == 'Male']
female_df = df[df['STUB_LABEL'] == 'Female']

plt.figure(figsize=(12, 9))
plt.plot(male_df['YEAR'], male_df['ESTIMATE'], color='blue', label='Male')
plt.plot(female_df['YEAR'], female_df['ESTIMATE'], color='red', label='Female')

plt.title('Rise in Suicides from 2010 to 2018 per 100,000')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()