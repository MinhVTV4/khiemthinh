
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { initializeApp } from "firebase/app";
import { getFirestore, doc, getDoc, collection, getDocs, setDoc } from "firebase/firestore";
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { MLClass } from '../types';

// Cấu hình Firebase từ yêu cầu của người dùng
const firebaseConfig = {
    apiKey: "AIzaSyCOofbN91yeOEWeAIP6wHgtGBfLZi9qJ1Y",
    authDomain: "h4rent-c12ef.firebaseapp.com",
    projectId: "h4rent-c12ef",
    storageBucket: "h4rent-c12ef.firebasestorage.app",
    messagingSenderId: "860501720608",
    appId: "1:860501720608:web:d6a81ae6aeba099c72c645"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

// Tên collection lưu trữ model
const COLLECTION_NAME = "sign_language_models";
const DOC_ID = "public_v1"; 
const SUB_COLLECTION_DATA = "data"; // Tên collection con chứa dữ liệu thô

export const loadFromFirestore = async (
    classifier: knnClassifier.KNNClassifier, 
    setClasses: (classes: MLClass[]) => void,
    setLastSaved: (date: Date) => void
) => {
    try {
        // 1. Tải Metadata (Danh sách lớp, version, timestamp)
        const docRef = doc(db, COLLECTION_NAME, DOC_ID);
        const docSnap = await getDoc(docRef);

        if (!docSnap.exists()) {
            console.log("No such document!");
            return false;
        }

        const metaData = docSnap.data();
        const { classes, timestamp, dataset: legacyDataset } = metaData;

        const tensorObj: {[key: string]: tf.Tensor} = {};

        // Hỗ trợ ngược: Nếu dữ liệu cũ nằm trong document chính (legacy)
        if (legacyDataset) {
            Object.keys(legacyDataset).forEach((key) => {
                const tData = legacyDataset[key];
                tensorObj[key] = tf.tensor(tData, [tData.length, tData[0].length]);
            });
        }

        // 2. Tải Dữ liệu chi tiết từ Sub-collection (Cách mới - Sharding)
        // Truy vấn vào sign_language_models/public_v1/data
        const dataCollectionRef = collection(db, COLLECTION_NAME, DOC_ID, SUB_COLLECTION_DATA);
        const dataSnapshot = await getDocs(dataCollectionRef);

        if (!dataSnapshot.empty) {
            dataSnapshot.forEach((doc) => {
                const classId = doc.id;
                const data = doc.data();
                
                if (data.samples && Array.isArray(data.samples) && data.samples.length > 0) {
                    // Merge hoặc ghi đè dữ liệu từ sub-collection
                    tensorObj[classId] = tf.tensor(data.samples, [data.samples.length, data.samples[0].length]);
                }
            });
        }

        if (Object.keys(tensorObj).length > 0) {
            classifier.clearAllClasses();
            classifier.setClassifierDataset(tensorObj);
            
            // Cập nhật danh sách lớp (tên, màu sắc...)
            if (classes) {
                setClasses(classes);
            }

            if (timestamp) {
                setLastSaved(new Date(timestamp));
            }

            return true;
        }

        return false;

    } catch (error) {
        console.error("Error loading from Firestore:", error);
        throw error;
    }
};

export const saveToFirestore = async (
    classifier: knnClassifier.KNNClassifier, 
    classes: MLClass[]
) => {
    try {
        const dataset = classifier.getClassifierDataset();
        
        // Note: Using Promise.all with individual setDoc calls instead of writeBatch
        // because writeBatch has a 10MB limit payload. Large ML datasets often exceed this,
        // causing "Unexpected state" or "Transaction too big" errors.
        
        // 1. Save Metadata (Classes info, timestamps) - Lightweight
        const metaRef = doc(db, COLLECTION_NAME, DOC_ID);
        const metaPayload = {
            classes: classes,
            timestamp: new Date().toISOString(),
            version: 5, 
            updatedBy: "admin",
            storageType: "sharded_firestore"
        };
        
        // We await this first to ensure the header is valid
        await setDoc(metaRef, metaPayload);

        // 2. Save Data Shards - Heavyweight
        const dataCollectionRef = collection(db, COLLECTION_NAME, DOC_ID, SUB_COLLECTION_DATA);
        
        const savePromises = Object.keys(dataset).map(async (classId) => {
            const data = dataset[classId].arraySync(); // Convert Tensor to Array
            const classDocRef = doc(dataCollectionRef, classId);
            
            // Each class data is saved as a separate document
            return setDoc(classDocRef, { 
                samples: data,
                updatedAt: new Date().toISOString()
            });
        });

        await Promise.all(savePromises);
        return true;

    } catch (error) {
        console.error("Error saving to Firestore:", error);
        throw error;
    }
};
