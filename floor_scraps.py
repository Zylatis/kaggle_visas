# File to keep code sippets I don't end up using but thought worth having


def kl_divergence( P, Q ):
	# Compute D_KL(P||Q)
	# Assumes they are evaluated on same x-grid
	# (could check in assert by forcing a 2D version to be fed in but is okay for now)
	assert len(P) == len(Q)

	Dkl = (np.array(P)*np.array(np.log(P/Q))).sum()

	return Dkl


# FROM THE PLOTTING ORDINAL FUNCTION
# This was to try look at similarity between histograms to see if certified and denied diff manifested
# in ordinals. Bit of overkill, just used ANOVA for mean for now
certified_density = ss.gaussian_kde(certified_data)
certified_density.covariance_factor = lambda : certified_kernel_factor
certified_density._compute_covariance()

denied_density = ss.gaussian_kde(denied_data)
denied_density.covariance_factor = lambda : denied_kernel_factor
denied_density._compute_covariance()

xs = np.arange(0,x_max,x_max/100.)
ys_certified = certified_density(xs)
ys_denied = denied_density(xs)

Dkl =  kl_divergence(ys_certified,ys_denied)

plt.plot(xs,ys_certified,antialiased=True, linewidth=2)
plt.plot(xs,ys_denied,antialiased=True, linewidth=2)

plt.xlim(right = x_max)
plt.xlim(left = 0) # assumes we won't be plottin negative stuff for now
plt.title(col + " split for certified (red) and denied (blue) - D_KL = " + str(round(Dkl,2)), fontdict = {'fontsize':10,'weight': 'bold'})
plt.xlabel(col,fontsize = 11)
plt.ylabel("PDF",fontsize = 11)
plt.savefig( img_folder + col + "_split_gaussian_kernel.png",dpi = 400)
plt.clf()