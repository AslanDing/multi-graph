import torch.nn.functional as F
import torch
from info_nce import info_nce

def nll_loss(output, target):
    return F.nll_loss(output, target)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

# mi loss
def mi_loss(query,pos_sample,neg_samples,pass_i):
    query = normalize(query)[0]
    pos_sample = normalize(pos_sample)[0]
    neg_samples = normalize(*neg_samples)

    if query.shape[0]<1:
        return 0
    if pos_sample.shape[0]<1:
        return 0
    if len(neg_samples)<=1:
        return 0

    sum_k_1 = 0
    count = 0
    for index , sample in enumerate(neg_samples):
        if index == pass_i:
            continue
        if sample.shape[0]<1:
            continue
        sum_k_1 += torch.exp(query@sample.transpose(1,0)).sum()
        count += 1
    if sum_k_1 == 0 or count==0:
        return 0

    sum_k_1 /= count

    result = torch.log( torch.exp(query@pos_sample.transpose(1,0)).sum()/sum_k_1 )
    if torch.isinf(result) or torch.isnan(result):
        print(" inf or nan ")
        return 0
    return result

def incidence(edge):
    '''
    get incidence matrix
    '''
    #E = nx.incidence_matrix(g1, oriented=True)
    E = edge.todense()
    return E

def vn_entropy(k, eps=1e-8):

    k = k/torch.trace(k)  # normalization

    #eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    try:
        eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    except Exception:
        print("error torch.linalg.eigvalsh ")
        return 0
    entropy = - torch.sum(eigv[eigv>0]*torch.log(eigv[eigv>0]+eps))
    #print("entropy ", entropy.cpu().detach().item())
    # if torch.abs(entropy)<0.5:
    #     return 0
    return entropy

def entropy_loss(sigma, rho, beta, alpha, entropy_fn = vn_entropy):
    assert(beta>=0), "beta shall be >=0"
    # sigma_dim=[]
    # for i in range(sigma.shape[0]):
    #     sigma_dim.append(sigma)

    sigma_diag = torch.diag(sigma)

    if sigma_diag.sum()==0:
        return 0

    connectivity_loss = alpha * torch.sum(torch.log(sigma_diag[sigma_diag>0]))
    #print(connectivity_loss)

    # l = entropy_fn(sigma)
    if beta>0:
        loss = 0.5*(1-beta)/beta * entropy_fn(sigma) + entropy_fn(0.5 * (sigma+rho))
        return loss - connectivity_loss
    else:
        return entropy_fn(sigma) - connectivity_loss
