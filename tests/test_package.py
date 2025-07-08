import directcov as m
import numpy as np

def test():
    indices = [
        [3, 2],
        [3, 3],
        [4, 0],
        [1, 2],
        [2, 2],
        [2, 2],
        [3, 2]
    ]
    values = [
        1.0,
        1.5,
        2.0,
        0.1,
        1.1,
        1.2,
        0.972
    ]
    event_ids = [
        0, 
        1,
        1,
        1,
        2,
        9,
        9
    ]

    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=np.float64)
    event_ids = np.asarray(event_ids, dtype=np.int32)

    print("Constructing")
    cov = m.DirectCov([5, 5])
    assert cov.shape() == [5,5]
    assert cov.strides() == [5, 1]
    assert cov.size() == 25
    assert np.array(cov, copy=False).sum() == 0.0
    print("\tpassed assertions")

    print("fillEvents()")
    cov.fillEvents(indices, values, event_ids)
    cov.finalize()
    
    #manually figure out what it should be...
    targetcov = np.zeros((5,5,5,5), dtype=np.float64)
    
    #event 0
    targetvec = np.zeros((5,5), dtype=np.float64)
    targetindices = np.asarray(indices)
    np.add.at(targetvec, (targetindices[:1,0], targetindices[:1,1]), values[:1])
    targetcov += np.einsum('ij,ab->ijab', targetvec, targetvec)
    #event 1
    targetvec = np.zeros((5,5), dtype=np.float64)
    np.add.at(targetvec, (targetindices[1:4,0], targetindices[1:4,1]), values[1:4])
    targetcov += np.einsum('ij,ab->ijab', targetvec, targetvec)
    #event 2
    targetvec = np.zeros((5,5), dtype=np.float64)
    np.add.at(targetvec, (targetindices[4:5,0], targetindices[4:5,1]), values[4:5])
    targetcov += np.einsum('ij,ab->ijab', targetvec, targetvec)
    #event 9
    targetvec = np.zeros((5,5), dtype=np.float64)
    np.add.at(targetvec, (targetindices[5:,0], targetindices[5:,1]), values[5:])
    targetcov += np.einsum('ij,ab->ijab', targetvec, targetvec)
    assert np.allclose(cov, targetcov)
    
    cov.fillEvents(indices, values, event_ids)
    cov.fillEvents(indices, values, event_ids)
    cov.finalize()
    assert np.allclose(cov, 3*targetcov)

    cov.fillEvents(indices, values, event_ids)
    cov.fillEvents(np.asarray([[0,0]]), np.asarray([10.]), np.asarray([9]))
    cov.finalize()
    
    prev = np.einsum('ij,ab->ijab', targetvec, targetvec)
    targetvec[0,0] += 10
    new = np.einsum('ij,ab->ijab', targetvec, targetvec)

    assert np.allclose(cov, 4*targetcov + (new-prev))
    print("\tpassed assertions")
