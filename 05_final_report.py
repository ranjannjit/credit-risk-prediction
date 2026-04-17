# %% [markdown]
# ## 8) Save a final report

# %%
with open('final_notes.txt', 'w') as f:
    f.write(results_df.to_string(index=False))

print('Done. Files created: model_comparison.csv, plots, and final_notes.txt')
'''

Path('output/credit_card_fraud_full_project_notebook.py').write_text(script)

# Split files
parts = {
'01_data_preprocessing.py': script.split('# %% [markdown]\n# ## 4) Traditional Machine Learning Models')[0],
'02_traditional_ml.py': '\n'.join(script.split('# %% [markdown]\n# ## 4) Traditional Machine Learning Models')[1].split('# %% [markdown]\n# ## 5) PyTorch datasets and models')[0]),
'03_pytorch_models.py': '\n'.join(script.split('# %% [markdown]\n# ## 5) PyTorch datasets and models')[1].split('# %% [markdown]\n# ## 6) Autoencoder anomaly detection')[0]),
'04_autoencoder_and_compare.py': '\n'.join(script.split('# %% [markdown]\n# ## 6) Autoencoder anomaly detection')[1])
}
for fn, txt in parts.items():
    Path('output/'+fn).write_text(txt)

print('created')