# !/bin/sh


echo "converting biobert_v1.0_pubmed_pmc weights"
tar -xzf biobert_v1.0_pubmed_pmc.tar.gz
transformers-cli convert --model_type bert \
  --tf_checkpoint biobert_v1.0_pubmed_pmc/biobert_model.ckpt \
  --config biobert_v1.0_pubmed_pmc/bert_config.json \
  --pytorch_dump_output biobert_v1.0_pubmed_pmc/pytorch_model.bin
echo "Done with converting"

echo "converting biobert_v1.0_pmc weights"
tar -xzf biobert_v1.0_pmc.tar.gz
transformers-cli convert --model_type bert \
  --tf_checkpoint biobert_v1.0_pmc/biobert_model.ckpt \
  --config biobert_v1.0_pmc/bert_config.json \
  --pytorch_dump_output biobert_v1.0_pmc/pytorch_model.bin
echo "Done with converting"

echo "converting biobert_v1.0_pubmed weights"
tar -xzf biobert_v1.0_pubmed.tar.gz
transformers-cli convert --model_type bert \
  --tf_checkpoint biobert_v1.0_pubmed/biobert_model.ckpt \
  --config biobert_v1.0_pubmed/bert_config.json \
  --pytorch_dump_output biobert_v1.0_pubmed/pytorch_model.bin
echo "Done with converting"

echo "converting biobert_v1.1_pubmed"
tar -xzf biobert_v1.1_pubmed.tar.gz
transformers-cli convert --model_type bert \
  --tf_checkpoint biobert_v1.1_pubmed/model.ckpt-1000000 \
  --config biobert_v1.1_pubmed/bert_config.json \
  --pytorch_dump_output biobert_v1.1_pubmed/pytorch_model.bin
echo "Done with converting";


