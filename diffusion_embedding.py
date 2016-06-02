class Diffusion_Embedding(object):

    def __init__(self,
                 source_path,
                 file_template,
                 subjects,
                 output_path,
                 diff_time = 0,
                 diff_alpha = 0.5,
                 diff_ncomp = 10,
                 subjects_subset = None,
                 output_suffix = 'embedding',
                 ftype = 'npy_timeseries', 
                 surf = 'fsaverage4',
                 mwall = False,
                 tp = None,
                 affinity_metric = 'correlation',
                 realign_method = 'STATIS'):

        """
        source_path : path with source timeseries/matrices, string
        file_template : template name for the files with timeseries, string
        subjects : the list of subjects to be inserted into file_template, list
        output_path : where to output the results, string
        diff_time : Diffusion time for individual embeddings, float
        diff_alpha : Value of diffusion operator, float
        diff_ncomp : Number of components to extract, int
        subjects_subset : subjects from which the template should be created, list (optional)
        output_prefix : prefix for output files, string
        ftype : type of source files (npy), string
        surf : surface for medial wall removal, string
        mwall : is medial wall present in source?, bool
        tp : timepoints to extract from source, list of arrays
        affinity_metric : affinity metric to use between timeseries
        """

        import numpy as np
        import os
        import glob

        self.source_path = source_path
        self.file_template = file_template
        self.subjects = subjects

        if subjects_subset == None:
            self.subjects_subset = subjects
        else:
            self.subjects_subset = subjects_subset

        self.ftype = ftype
        self.surf = surf
        self.mwall = mwall
        self.tp = tp
        self.affinity_metric = affinity_metric
        self.output_suffix = output_suffix
        self.output_path = output_path
        self.diff_time = diff_time
        self.diff_alpha = diff_alpha
        self.diff_ncomp = diff_ncomp

        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)

    def get_source_files(self):

        # Get the list of input files and store it

        import os

        self.input_files_ = [os.path.join(self.source_path, self.file_template % s) for s in self.subjects]
        self.template_subjects_ = [s for s in self.subjects_subset if s in self.subjects]

    def load_data(self, index):

        # Load data and optionally select timepoints from indicated source timeseries

        import numpy as np

        # Load data
        if self.ftype == 'npy_timeseries' or self.ftype == 'npy_matrix':
            data = np.load(self.input_files_[index])

        # Select timepoints if neccessary

        if self.tp == None:
            data_sel = data
        elif len(self.tp) == 1:
            data_sel = data[:,tp]
        elif len(self.tp) > 1:
            data_sel = data[:,tp[index]]

        # Select rows if desired
        if self.mwall == True:
            self.remove_medial_wall(data_sel)
        elif self.mwall == False:
            self.current_data_ = data_sel

    def remove_medial_wall(self, data_sel):

        # Remove the medial wall vertices based on provided surface shape

        import os
        import nibabel as nib
        import numpy as np

        fspath = os.environ.get('FREESURFER_HOME')
        nv = nib.freesurfer.io.read_geometry(os.path.join(fspath, 'subjects', self.surf, 'surf', 'lh.pial'))[0].shape[0]
        lhcort = np.sort(nib.freesurfer.io.read_label(os.path.join(fspath, 'subjects', self.surf, 'label', 'lh.cortex.label')))
        rhcort = np.sort(nib.freesurfer.io.read_label(os.path.join(fspath, 'subjects', self.surf, 'label', 'rh.cortex.label')))+nv

        cortex = np.hstack([lhcort,rhcort])

        self.current_data_ = data_sel[cortex,:]

    def check_data(self):

        # Check if the timeseries makes sense, i.e. does not contain NaNs or Inf

        import numpy as np

        nancount = np.sum(np.isnan(self.current_data_))
        infcount = np.sum(np.isinf(self.current_data_))

        print "Found %d NaN and %d Inf values in the timeseries" % (nancount, infcount)

        if np.logical_and(nancount == 0, infcount == 0):
            self.current_ok_ = True
        else:
            self.current_ok_ = False

    def calculate_affinity(self):

        import numpy as np

        if not self.current_ok_:
            raise ValueError('There is something wrong with the timeseries, cannot proceed')

        if self.affinity_metric == 'correlation':
            self.current_cmat_ = np.corrcoef(self.current_data_)
            self.current_cmat_ = (self.current_cmat_ + 1) / 2

    def embed_affinity(self):

        import time
        from mapalign import embed

        stime = time.time()
        self.compute_diffusion_map()
        print "Diffusion embedding took %d seconds" % (time.time()-stime)

    def compute_embeddings(self):

        import os
        import numpy as np

        self.get_source_files()
        self.embedded_files_ = []

        for i, s in enumerate(self.subjects):

            f = os.path.join(self.output_path, s + '_' + self.output_suffix + '.npz')

            if os.path.isfile(f):
                print "Embedding already computed for subject %s, skipping" % s
                self.embedded_files_.append(f)
                continue

            self.load_data(i)
            self.check_data()

            if self.ftype == 'npy_timeseries':
                self.calculate_affinity()

            self.embed_affinity()

            np.savez(f, self.current_res_)
            self.embedded_files_.append(f)

        if len(self.embedded_files_) == len(self.subjects):
            self.embedding_complete_ = True
            self.template_files_ = [os.path.join(self.output_path, s + '_' + self.output_suffix + '.npz') for s in self.template_subjects_]
        else:
            self.embedding_complete_ = False

    def realign_embeddings(self, filelist = None):

        from pySTATIS import statis
        import numpy as np
        import os

        if filelist is not None:
            self.template_files_ = filelist

        self.X_ev_ = []
        self.X_em_ = []

        print "Getting data for STATIS..."

        for i, f in enumerate(self.template_files_):
            t = np.load(f)['arr_0'].item()
            n = t['vectors'][:,0]
            ev = (t['vectors'].T/n).T
            self.X_ev_.append(ev[:,1:11])
            self.X_em_.append( (ev*t['orig_lambdas'])[:,1:11] )

        print "Running STATIS..."

        self.statis_ = statis.statis(self.X_ev_, self.subjects_subset, os.path.join(self.output_path,'statis_results.npy'))

    def project_template_subjects(self):
        """
        Create projections of individual embeddings onto the template.
        This function is for subjects who participated in the template creation process.
        """

        from pySTATIS import statis
        import os

        self.projection_path_ = os.path.join(self.output_path, 'projections')

        if not os.path.isdir(self.projection_path_):
            os.mkdir(self.projection_path_)

        statis.project_back(self.X_em_, self.statis_['Q'], self.projection_path_, self.subjects_subset)

    def compute_markov_matrix(self, skip_checks=False, overwrite=False):

        """
        Slightly modified code originally written by Satrajit Ghosh (satra@mit.edu github.com/satra/mapalign)
        """

        import numpy as np
        import scipy.sparse as sps

        L = self.current_cmat_
        alpha = self.diff_alpha

        use_sparse = False
        if sps.issparse(L):
            use_sparse = True

        if not skip_checks:
            from sklearn.manifold.spectral_embedding_ import _graph_is_connected
            if not _graph_is_connected(L):
                raise ValueError('Graph is disconnected')

        ndim = L.shape[0]
        if overwrite:
            L_alpha = L
        else:
            L_alpha = L.copy()

        if alpha > 0:
            # Step 2
            d = np.array(L_alpha.sum(axis=1)).flatten()
            d_alpha = np.power(d, -alpha)
            if use_sparse:
                L_alpha.data *= d_alpha[L_alpha.indices]
                L_alpha = sps.csr_matrix(L_alpha.transpose().toarray())
                L_alpha.data *= d_alpha[L_alpha.indices]
                L_alpha = sps.csr_matrix(L_alpha.transpose().toarray())
            else:
                L_alpha = d_alpha[:, np.newaxis] * L_alpha
                L_alpha = L_alpha * d_alpha[np.newaxis, :]

        # Step 3
        d_alpha = np.power(np.array(L_alpha.sum(axis=1)).flatten(), -1)
        if use_sparse:
            L_alpha.data *= d_alpha[L_alpha.indices]
        else:
            L_alpha = d_alpha[:, np.newaxis] * L_alpha

        return L_alpha

    def compute_diffusion_map(self, skip_checks=False, overwrite=False):
        """
        Slightly modified code originally written by Satrajit Ghosh (satra@mit.edu github.com/satra/mapalign)

        Compute the diffusion maps of a symmetric similarity matrix
            L : matrix N x N
               L is symmetric and L(x, y) >= 0
            alpha: float [0, 1]
                Setting alpha=1 and the diffusion operator approximates the
                Laplace-Beltrami operator. We then recover the Riemannian geometry
                of the data set regardless of the distribution of the points. To
                describe the long-term behavior of the point distribution of a
                system of stochastic differential equations, we can use alpha=0.5
                and the resulting Markov chain approximates the Fokker-Planck
                diffusion. With alpha=0, it reduces to the classical graph Laplacian
                normalization.
            n_components: int
                The number of diffusion map components to return. Due to the
                spectrum decay of the eigenvalues, only a few terms are necessary to
                achieve a given relative accuracy in the sum M^t.
            diffusion_time: float >= 0
                use the diffusion_time (t) step transition matrix M^t
                t not only serves as a time parameter, but also has the dual role of
                scale parameter. One of the main ideas of diffusion framework is
                that running the chain forward in time (taking larger and larger
                powers of M) reveals the geometric structure of X at larger and
                larger scales (the diffusion process).
                t = 0 empirically provides a reasonable balance from a clustering
                perspective. Specifically, the notion of a cluster in the data set
                is quantified as a region in which the probability of escaping this
                region is low (within a certain time t).
            skip_checks: bool
                Avoid expensive pre-checks on input data. The caller has to make
                sure that input data is valid or results will be undefined.
            overwrite: bool
                Optimize memory usage by re-using input matrix L as scratch space.
            References
            ----------
            [1] https://en.wikipedia.org/wiki/Diffusion_map
            [2] Coifman, R.R.; S. Lafon. (2006). "Diffusion maps". Applied and
            Computational Harmonic Analysis 21: 5-30. doi:10.1016/j.acha.2006.04.006
        """

        M = self.compute_markov_matrix(skip_checks, overwrite)

        from scipy.sparse.linalg import eigsh, eigs
        import numpy as np

        ndim = self.current_cmat_.shape[0]

        # Step 4
        func = eigs
        if self.diff_ncomp is not None:
            lambdas, vectors = func(M, k=self.diff_ncomp + 1)
        else:
            lambdas, vectors = func(M, k=max(2, int(np.sqrt(ndim))))
        del M

        if func == eigsh:
            lambdas = lambdas[::-1]
            vectors = vectors[:, ::-1]
        else:
            lambdas = np.real(lambdas)
            vectors = np.real(vectors)
            lambda_idx = np.argsort(lambdas)[::-1]
            lambdas = lambdas[lambda_idx]
            vectors = vectors[:, lambda_idx]

        # Step 5

        psi = vectors/vectors[:, [0]]
        olambdas = lambdas.copy()

        if self.diff_time == 0:
            lambdas = lambdas[1:] / (1 - lambdas[1:])
        else:
            lambdas = lambdas[1:] ** float(diffusion_time)
        lambda_ratio = lambdas/lambdas[0]
        threshold = max(0.05, lambda_ratio[-1])

        n_components_auto = np.amax(np.nonzero(lambda_ratio > threshold)[0])
        n_components_auto = min(n_components_auto, ndim)
        if self.diff_ncomp is None:
            self.diff_ncomp = n_components_auto
        self.current_emb_ = psi[:, 1:(self.diff_ncomp + 1)] * lambdas[:self.diff_ncomp][None, :]

        self.current_res_ = dict(lambdas=lambdas, orig_lambdas = olambdas, vectors=vectors,
                      n_components=self.diff_ncomp, diffusion_time=self.diff_time,
                      n_components_auto=n_components_auto)
