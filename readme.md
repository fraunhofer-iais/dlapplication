# Branching model:

master - 1 - the master branch

hotfix - many - hotfixes to the current master, branched from master

release - 1- preparation of the next release. As soon as a stable release is built, this branch is merged into master

development - 1 - the current development branch. Selected results are merged into the release branch

feature - many - feature development, branched from development. Its results are merged into the development branch.

experiment - many - branch to make experiments. It is kept to ensure reproducibility. Usually branched from master, but can also be branched from development. Usually not merged back. 


# Copyright
Copyright 2020 Fraunhofer IAIS

Core contributors:

Michael Kamp info@michaelkamp.org

Linara Adilova linara.adilova@iais.fraunhofer.de

Further contributors:

Tim Wirtz tim.tirtz@iais.fraunhofer.de

Joachim Sicking joachim.sicking@iais.fraunhofer.de

Nathalie Paul nathalie.paul@iais.fraunhofer.de

Alexander Zimmerman alexander.zimmermann@iais.fraunhofer.de
