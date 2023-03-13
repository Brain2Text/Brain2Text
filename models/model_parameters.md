| Type of hyperparameter                   || Hyperparameter           |
|-----------------------|--------------------|-------------------------|
| Embedding vector from EEG | Features           | 104                     |
|                       | Time points        | 16                      |
| Generator-pre/mid/postNet | Out channel        | 512 / 1024 / 80         |
| ^                      | Conv1D kernel size | 3 / 3 / 9               |
| ^                      | Dilation           | 1 / 1 / 1                |
| ^                      | Padding            | 1 / 0 / 4               |
| Generator-GRU         | Out channel        | 256                     |
|                       | Layers             | 1                       |
|                       | direction          | Bi-directional          |
| Generator-Upblock     | The number of blocks | 3                       |
|                       | Out channel        | 512 / 256 / 128         |
|                       | Upsample rate      | 3 / 2 / 2               |
|                       | Upsample kernel size | 6 / 4 / 4               |
|                       | Padding            | 1 / 1 / 1               |
| Generator-Resblock    | The number of blocks | 3                       |
|                       | Out channel        | 512 / 256 / 128         |
|                       | Conv1D kernel      | 3 / 7 / 11              |
|                       | Dilation           | 1,3,5 / 1,3,5 / 1,3,5   |
|                       | Padding            | 1,3,5 / 3,9,15 / 5,15,25 |
| Discriminator-preNet  | Out channel        | 64                      |
|                       | Conv1D kernel size | 3                       |
|                       | Dilation           | 1                       |
|                       | Padding            | 1                       |
| Discriminator-Downblock | The number of blocks | 3                      |
|                       | Out channel        | 128 / 256 / 512        |
|                       | Upsample rate      | 3 / 3 / 3              |
|                       | Upsample kernel size | 6 / 6 / 6              |
|                       | Padding            | 1 / 1 / 1              |
| Discriminator-Resblock | The number of blocks | 3                      |
|                       | Out channel        | 128 / 256 / 512        |
|                       | Conv1D kernel      | 11 / 7 / 3             |
|                       | Dilation           | 1,3,5 / 1,3,5 / 1,3,5  |
|                       | Padding            | 5,15,25 / 3,9,15 / 1,3,5 |
| Discriminator-GRU     | Out channel        | 256                     |
|                       | Layers             | 1                       |
|                       | direction          | Bi-directional          |
