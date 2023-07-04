# Document Automation

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 1.16.0](https://img.shields.io/badge/AppVersion-1.16.0-informational?style=flat-square)

A Helm chart for Kubernetes

## Values

| Key                             | Type   | Default                 | Description                                                                                                        |
|---------------------------------|--------|-------------------------|--------------------------------------------------------------------------------------------------------------------|
| dataset.local.dataset_path      | string | `"nil"`                 | Host Path to input dataset                                                                                         |
| dataset.local.docvqa_path       | string | `"nil"`                 | Host Path to docvqa dataset                                                                                        |
| dataset.local.es_data_path      | string | `"nil"`                 | Host Path to store es data                                                                                         |
| dataset.local.output_path       | string | `"nil"`                 | Host Path to store output                                                                                          |
| dataset.local.postgre_data_path | string | `"nil"`                 | Host Path to store postgres data                                                                                   |
| dataset.local.config_path       | string | `"nil"`                 | Host Path to Config Directory                                                                                      |
| dataset.nfs.dataset_path        | string | `"nil"`                 | Path to input dataset                                                                                              |
| dataset.nfs.docvqa_path         | string | `"nil"`                 | Path to docvqa dataset                                                                                             |
| dataset.nfs.config_path         | string | `"nil"`                 | Path to Config in Local NFS                                                                                        |
| dataset.nfs.path                | string | `"nil"`                 | Path of NFS                                                                                                        |
| dataset.nfs.server              | string | `"nil"`                 | Hostname of NFS Server                                                                                             |
| dataset.type                    | string | `"<local/nfs>"`         | `local,nfs` dataset input enabler                                                                                  |
| image.name                      | string | `"intel/ai-workflows"`  |                                                                                                                    |
| metadata.name                   | string | `"document-automation"` |                                                                                                                    |
| model_name                      | string | `"my_dpr_model"`        |                                                                                                                    |
| proxy                           | string | `"nil"`                 |                                                                                                                    |
| pvc.storage                     | string | `"15Gi"`                | Size of PVC storage, if the default value cannot meet the requirements, please adjust as needed                    |
| retrieval_method                | string | `"ensemble"`            | Support `ensemble,bm25,dpr`                                                                                        |