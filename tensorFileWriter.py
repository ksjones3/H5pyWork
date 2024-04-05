#!/usr/bin/env python

import tensorflow as tf
import uproot
import numpy as np
import sys
import h5py
import awkward as ak


def openRootFile(filePath):
    rootFile = uproot.open(filePath)
    tree = rootFile['analysis']
    branches = tree.arrays()
    rootFile.close()
    return branches

def makeConstantsTensor(branches):
    cell_x = np.array(branches['cell_x'][0])
    cell_y = np.array(branches['cell_y'][0])
    cell_z = np.array(branches['cell_z'][0])
    cell_eta = np.array(branches['cell_eta'][0])
    cell_phi = np.array(branches['cell_phi'][0])
    cell_subCalo = np.array(branches['cell_subCalo'][0])
    cell_hashID = np.array(branches['cell_hashID'][0])
    cell_noiseSigma = np.array(branches['cell_noiseSigma'][0])
    length = len(cell_x)
    constantsData = np.zeros((length, 8))
    
    constantsData[:, 0] = cell_x
    constantsData[:, 1] = cell_y
    constantsData[:, 2] = cell_z
    constantsData[:, 3] = cell_eta
    constantsData[:, 4] = cell_phi
    constantsData[:, 5] = cell_subCalo
    constantsData[:, 6] = cell_hashID
    constantsData[:, 7] = cell_noiseSigma
    
    constantsTensor = tf.constant(constantsData)
    print("Completed Constants Tensor")
    return constantsTensor

def makeCellDataTensor(branches):
    cell_sampling = np.array(branches['cell_sampling'])
    cell_e = np.array(branches['cell_e'])
    cell_time = np.array(branches['cell_time'])
    cell_weight = np.array(branches['cell_weight'])
    cell_truth = np.array(branches['cell_truth'])
    cell_cluster_index = np.array(branches['cell_cluster_index'])
    cell_to_cluster_e = np.array(branches['cell_to_cluster_e'])
    cell_to_cluster_eta = np.array(branches['cell_to_cluster_eta'])
    cell_to_cluster_phi = np.array(branches['cell_to_cluster_phi'])
    cell_SNR = np.array(branches['cell_SNR'])
    cell_cluster_index = np.array(branches['cell_cluster_index'])
    cell_to_cluster_e = np.array(branches['cell_to_cluster_e'])
    cell_to_cluster_eta = np.array(branches['cell_to_cluster_eta'])
    cell_to_cluster_phi = np.array(branches['cell_to_cluster_phi'])
    
    number_of_events = len(cell_e)
    number_of_cells = len(cell_e[0])
    cellData = np.zeros((number_of_events, number_of_cells, 10))

    cellData[:, :, 0] = cell_sampling
    cellData[:, :, 1] = cell_e
    cellData[:, :, 2] = cell_time
    cellData[:, :, 3] = cell_weight
    cellData[:, :, 4] = cell_truth
    cellData[:, :, 5] = cell_cluster_index
    cellData[:, :, 6] = cell_to_cluster_e
    cellData[:, :, 7] = cell_to_cluster_eta
    cellData[:, :, 8] = cell_to_cluster_phi
    cellData[:, :, 9] = cell_SNR
    
    
    cellDataTensor = tf.constant(cellData)
    print("Completed Cell Data Tensor")
    return cellDataTensor

def makeEventDataTensor(branches):
    EventNumber = np.array(branches["EventNumber"])
    
    eventDataTensor = tf.constant(EventNumber)
    print("Completed Event Data Tensor")
    return eventDataTensor

def findMaxClusterNumber(branches):
    maxClusterNumber = 0
    clusterEnergyEvents = branches["cluster_e"]
    for clusterEnergy in clusterEnergyEvents:
        clusterNumber = len(clusterEnergy)
        if clusterNumber > maxClusterNumber:
            maxClusterNumber = clusterNumber
    return maxClusterNumber

def zeroPackData(data, maxClusterNumber):
    zeroPackedData = []
    for event in data:
        length = len(event)
        if length < maxClusterNumber:
            numberOfZeroes = maxClusterNumber-length
            zeroArray = ak.zeros_like(np.zeros((numberOfZeroes)))
            zeroPackedEvent = ak.concatenate((event, zeroArray))
        else:
            zeroPackedEvent = event
        zeroPackedData.append(zeroPackedEvent)
    return np.array(zeroPackedData)

def makeClusterDataTensor(branches):
    maxClusterNumber = findMaxClusterNumber(branches)
    
    cluster_e = zeroPackData(branches['cluster_e'], maxClusterNumber)
    cluster_eta = zeroPackData(branches['cluster_eta'], maxClusterNumber)
    cluster_phi = zeroPackData(branches['cluster_phi'], maxClusterNumber)
    
    
    number_of_events = len(cluster_e)
    clusterData = np.zeros((number_of_events, maxClusterNumber, 3))

    clusterData[:, :, 0] = cluster_e
    clusterData[:, :, 1] = cluster_eta
    clusterData[:, :, 2] = cluster_phi
    
    clusterDataTensor = tf.constant(clusterData)
    print("Completed Cluster Data Tensor")
    return clusterDataTensor

def writeToh5File(h5File, constantTensor, cellDataTensor, eventDataTensor, clusterDataTensor, counter):
    h5File.create_dataset("Constants"+str(counter), data = constantTensor, compression="gzip")
    h5File.create_dataset("Cell_Data"+str(counter), data = cellDataTensor, compression="gzip")
    h5File.create_dataset("Event_Data"+str(counter), data = eventDataTensor, compression="gzip")
    h5File.create_dataset("Cluster_Data"+str(counter), data = clusterDataTensor, compression="gzip")

def readData(fullData):
    constants = makeConstantsTensor(fullData)
    cellData = makeCellDataTensor(fullData)
    eventData = makeEventDataTensor(fullData)
    clusterData = makeClusterDataTensor(fullData)
    return constants, cellData, eventData, clusterData

def createOutputFile(inputFilePaths, outputFile):
    h5File = h5py.File(outputFile, 'w')
    counter = 1
    for f in inputFilePaths:
        data = openRootFile(f)
        constants, cellData, eventData, clusterData = readData(data)
        writeToh5File(h5File, constants, cellData, eventData, clusterData, counter)
        counter = counter+1
    h5File.close()

outputFilePath = sys.argv[1]
inputFilePaths = sys.argv[2:]

createOutputFile(inputFilePaths, outputFilePath)
