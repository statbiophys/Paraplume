This is the folder where comparisons with other methods were made.
- We used the **mipe**, **pecan** and **paragraph** datasets.
- In the `Scripts` folder, bash scripts starting with `create_datasets` were used to create labeled datasets. It takes as input a csv files present in the `dataset` folder and a folder of 3D structures from SabDab.
- The structures used for the paragraph dataset are in the `structures/imgt_renumbered_expanded` folder, structures for the pecan dataset in `structures/imgt_renumbered_pecan`, and for mipe dataset, in `structures/imgt_renumbered_mipe`. For the mipe dataset we also modeled structures using ABB3 to re-benchmark Paragraph without experimentally solved structures as inputs on their dataset. This is because we found that the results displayed for Paragraph in the MIPE paper were surprisingly low. All structures were IMGT renumbered to correct numbering errors.
- The `create_dataset` scripts create two files: a `.json` file with the sequences and their paratope labels, and a `.pt` file with the embeddings of each sequence.
- Script `train_mipe.sh` (resp. `train_paragraph.sh` and `train_pecan.sh`) was used to train Paraplume on the mipe (resp. paragraph** and pecan) dataset. Script `train_mipe_3D.sh` (resp. pecan and paragraph) trains Paragraph on the mipe (resp. paragraph and pecan) dataset, which was needed to create Paraplume-G, the combination of Paraplume and Paragraph.
- Scripts starting by `chain_study` were used to test the trained models on single chains only. Results are saved in the `mipe`, `paragraph` and `pecan` folders.
- In `6b0s` we analyze the predictions of Paraplume trained on the Paragraph train set on an unseen sequence in the test set: antibody with PDB code 6b0s. We used PyMol to compare the predictions with the ground truth.
- Benchmark for Paraplume was done using `benchmark/paraplume_requirements.txt`
- Benchmark for Paraplume-G was done using `benchmark/paraplume_G_requirements.txt`

⚠️ **Note**: If reproducing the results, some of the folder/file paths might need changing. Contact gabrielathenes@gmail.com if you need help.
