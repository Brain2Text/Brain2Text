<table>
  <caption>Hyperparameter of NeuroTalk models in details</caption>
  <thead>
    <tr>
      <th colspan="2">Type of hyperparameter</th>
      <th>Hyperparameter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Embedding vector from EEG</td>
      <td>Features</td>
      <td>104</td>
    </tr>
    <tr>
      <td>Time points</td>
      <td>16</td>
    </tr>
    <tr>
      <td rowspan="4">Generator-pre/mid/postNet</td>
      <td>Out channel</td>
      <td>512 / 1024 / 80</td>
    </tr>
    <tr>
      <td>Conv1D kernel size</td>
      <td>3 / 3 / 9</td>
    </tr>
    <tr>
      <td>Dilation</td>
      <td>1 / 1 / 1</td>
    </tr>
    <tr>
      <td>Padding</td>
      <td>1 / 0 / 4</td>
    </tr>
    <tr>
      <td rowspan="3">Generator-GRU</td>
      <td>Out channel</td>
      <td>256</td>
    </tr>
    <tr>
      <td>Layers</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Direction</td>
      <td>Bi-directional</td>
    </tr>
    <tr>
      <td rowspan="5">Generator-Upblock</td>
      <td>The number of blocks</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Out channel</td>
      <td>512 / 256 / 128</td>
    </tr>
    <tr>
      <td>Upsample rate</td>
      <td>3 / 2 / 2</td>
    </tr>
    <tr>
      <td>Upsample kernel size</td>
      <td>6 / 4 / 4</td>
    </tr>
    <tr>
      <td>Padding</td>
      <td>1 / 1 / 1</td>
    </tr>
    <tr>
      <td rowspan="5">Generator-Resblock</td>
      <td>The number of blocks</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Out channel</td>
      <td>512 / 256 / 128</td>
    </tr>
    <tr>
      <td>Conv1D kernel</td>
      <td>3 / 7 / 11</td>
    </tr>
    <tr>
      <td>Dilation</td>
      <td>1,3,5 / 1,3,5 / 1,3,5</td>
    </tr>
    <tr>
      <td>Padding</td>
      <td>1,3,5 / 3,9,15 / 5,15,25</td>
    </tr>
    <tr>
      <td rowspan="4">Discriminator-preNet</td>
      <td>Out channel</td>
      <td>64</td>
    </tr>
    <tr>
      <td>Conv1D kernel size</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Dilation</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Padding</td>
      <td>1</td>
    </tr>
    <tr>
      <td rowspan="5">Discriminator-Downblock</td>
      <td>The number of blocks</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Out channel</td>
      <td>128 / 256 / 512</td>
    </tr>
    <tr>
      <td>Upsample rate</td>
      <td>3 / 3 / 3</td>
    </tr>
    <tr>
      <td>Upsample kernel size</td>
      <td>6 / 6 / 6</td>
    </tr>
    <tr>
      <td>Padding</td>
      <td>1 / 1 / 1</td>
    </tr>
    <tr>
      <td rowspan="5">Discriminator-Resblock</td>
      <td>The number of blocks</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Out channel</td>
      <td>128 / 256 / 512</td>
    </tr>
    <tr>
      <td>Conv1D kernel</td>
      <td>11 / 7 / 3</td>
    </tr>
    <tr>
      <td>Dilation</td>
      <td>1,3,5 / 1,3,5 / 1,3,5</td>
    </tr>
    <tr>
      <td>Padding</td>
      <td>5,15,25 / 3,9,15 / 1,3,5</td>
    </tr>
    <tr>
      <td rowspan="3">Discriminator-GRU</td>
      <td>Out channel</td>
      <td>256</td>
    </tr>
    <tr>
      <td>Layers</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Direction</td>
      <td>Bi-directional</td>
    </tr>
  </tbody>
</table>
