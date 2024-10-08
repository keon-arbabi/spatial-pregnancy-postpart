if isinstance(genes_to_label, (int, np.integer)):
    label = genes_to_label != 0
    if label:
        x_to_label = table['logFC'][genes_to_label]
        y_to_label = table[significance_column][genes_to_label]
        genes_to_label = table['gene'][genes_to_label]
elif label_kwargs is not None:
    error_message = (
        'label_kwargs cannot be specified when genes_to_label=0, '
        'since no genes are being labeled')
    raise ValueError(error_message)
elif genes_to_label is not None:
    label = True
    genes_to_label = \
        to_tuple_checked(genes_to_label, 'genes_to_label', str,
                         'strings')
    genes_to_label = pl.DataFrame({'gene': genes_to_label})\
        .join(table.select('gene', 'logFC', significance_column),
              how='left', on='gene')
    num_missing = genes_to_label['logFC'].null_count()
    if num_missing == len(genes_to_label):
        error_message = (
            "none of the specified genes were found in table['gene']")
        raise ValueError(error_message)
    elif num_missing > 0:
        gene = genes_to_label.filter(pl.col.logFC.is_null())['gene'][0]
        error_message = (
            f"one of the specified genes, {gene!r}, was not found in "
            "table['gene']")
        raise ValueError(error_message)
if label:
    print(genes_to_label)