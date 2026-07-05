# NightJet Model Weight License And Provenance Notes

The NightJet source code is MIT licensed. The model weights published in this
Hugging Face repository are not automatically covered by the MIT source license.

These weights are provided for research, demonstration, and evaluation use. They
were trained from local low-light camera data and targets distilled from
teacher-model outputs. Those upstream teacher models and datasets may carry
their own license, attribution, redistribution, or commercial-use obligations.

Do not redistribute, commercialize, or embed these weights in a product until
you have reviewed:

- the NightJet model card;
- the provenance fields in `manifest.json`;
- the license and terms for upstream teacher models used to create the targets;
- the rights associated with any camera data or examples you include.

TensorRT `.plan` and `.engine` files are target-specific runtime build outputs.
They are not canonical model artifacts and are not covered by this model repo.
