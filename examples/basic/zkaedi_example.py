"""
zkAEDI Example Usage
====================
Demonstrates the complete zkAEDI system with datasets and models
"""

from zkaedi.core.authenticated_encryption import create_zk_authenticated_encryption
from zkaedi.core.zk_primitives import create_zk_primitives
from zkaedi.models.zkaedi_model import ModelConfig, create_zkaedi_model, create_anomaly_detector
from zkaedi.datasets.synthetic_generator import DatasetConfig, create_dataset_generator
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add zkAEDI to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def demonstrate_dataset_generation():
    """Demonstrate synthetic dataset generation"""
    print("\n" + "="*60)
    print("zkAEDI Dataset Generation Demo")
    print("="*60)

    # Configure dataset
    config = DatasetConfig(
        name="financial_fraud_detection",
        num_samples=10000,
        num_features=20,
        noise_level=0.1,
        anomaly_rate=0.02,
        temporal=True,
        categorical_features=3,
        missing_rate=0.05,
        seed=42
    )

    # Create generator
    generator = create_dataset_generator(config)

    # Generate different types of datasets
    datasets = {}

    print("\nGenerating datasets...")
    for dataset_type in ['financial', 'healthcare', 'iot', 'cybersecurity']:
        print(f"  - Generating {dataset_type} dataset...")
        df = generator.generate_dataset(dataset_type)
        datasets[dataset_type] = df

        # Save dataset
        os.makedirs('datasets', exist_ok=True)
        filepath = f'datasets/{dataset_type}_synthetic.csv'
        generator.save_dataset(df, filepath)

        # Generate and save metadata
        metadata = generator.generate_metadata(df)
        import json
        with open(f'datasets/{dataset_type}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"    ✓ Saved to {filepath}")
        print(f"    ✓ Shape: {df.shape}")
        print(f"    ✓ Columns: {list(df.columns)[:5]}...")

    return datasets


def demonstrate_model_training(datasets):
    """Demonstrate zkAEDI model training with zero-knowledge proofs"""
    print("\n" + "="*60)
    print("zkAEDI Model Training Demo")
    print("="*60)

    # Use financial dataset for fraud detection
    df = datasets['financial']

    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in [
        'timestamp', 'is_anomaly']]
    X = df[feature_cols]
    y = df['is_anomaly'].astype(int)

    # Configure model
    model_config = ModelConfig(
        model_type="random_forest",
        n_estimators=100,
        max_depth=10,
        enable_zk_proofs=True,
        privacy_budget=1.0
    )

    # Create and train model
    print("\nTraining zkAEDI model with zero-knowledge proofs...")
    model = create_zkaedi_model(model_config)

    # Train model
    result = model.train(X, y)

    print(f"\nTraining Results:")
    print(f"  - Model ID: {result.model_id}")
    print(f"  - Accuracy: {result.accuracy:.4f}")
    print(f"  - Precision: {result.precision:.4f}")
    print(f"  - Recall: {result.recall:.4f}")
    print(f"  - F1 Score: {result.f1_score:.4f}")
    print(f"  - Training Time: {result.training_time:.2f}s")
    print(f"  - ZK Proof Generated: {result.zk_proof is not None}")

    # Verify model integrity
    if result.zk_proof:
        verified = model.verify_model_integrity(
            result.model_id, result.zk_proof)
        print(f"  - Model Integrity Verified: {verified}")

    # Make predictions with proof
    print("\nMaking predictions with zero-knowledge proof...")
    test_samples = X.sample(n=100, random_state=42)
    predictions, prediction_proof = model.predict_with_proof(test_samples)

    print(f"  - Predictions made: {len(predictions)}")
    print(f"  - Anomalies detected: {sum(predictions)}")
    print(f"  - Proof generated: {prediction_proof['timestamp']}")

    # Export encrypted model
    print("\nExporting encrypted model...")
    os.makedirs('models', exist_ok=True)
    export_path = 'models/fraud_detection_model_encrypted.json'
    export_data = model.export_model_encrypted(export_path)
    print(f"  - Model exported to: {export_path}")
    print(
        f"  - Encryption algorithm: {export_data['metadata']['encryption_algorithm']}")
    print(f"  - Model commitment: {export_data['commitment'][:32]}...")

    # Generate privacy report
    privacy_report = model.generate_privacy_report()
    print("\nPrivacy Report:")
    for key, value in privacy_report['security_features'].items():
        print(f"  - {key}: {value}")

    return model


def demonstrate_anomaly_detection(datasets):
    """Demonstrate zkAEDI anomaly detection"""
    print("\n" + "="*60)
    print("zkAEDI Anomaly Detection Demo")
    print("="*60)

    # Use IoT dataset for anomaly detection
    df = datasets['iot']

    # Prepare features
    feature_cols = [col for col in df.columns if col not in [
        'timestamp', 'is_anomaly', 'device_id']]
    X = df[feature_cols]

    # Create anomaly detector
    print("\nCreating zkAEDI anomaly detector...")
    detector = create_anomaly_detector()

    # Train on normal data only
    normal_data = df[~df['is_anomaly']]
    X_normal = normal_data[feature_cols]
    y_normal = np.zeros(len(X_normal))  # All normal samples

    print(f"Training on {len(X_normal)} normal samples...")
    training_result = detector.train(X_normal, y_normal)

    # Detect anomalies on full dataset
    print("\nDetecting anomalies...")
    anomalies, anomaly_report = detector.detect_anomalies(X)

    # Compare with ground truth
    true_anomalies = df['is_anomaly'].values
    detected_anomalies = anomalies.astype(bool)

    true_positives = sum(true_anomalies & detected_anomalies)
    false_positives = sum(~true_anomalies & detected_anomalies)
    false_negatives = sum(true_anomalies & ~detected_anomalies)
    true_negatives = sum(~true_anomalies & ~detected_anomalies)

    print(f"\nAnomaly Detection Results:")
    print(f"  - Total samples: {anomaly_report['total_samples']}")
    print(f"  - Anomalies detected: {anomaly_report['anomalies_detected']}")
    print(f"  - Anomaly rate: {anomaly_report['anomaly_rate']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  - True Positives: {true_positives}")
    print(f"  - False Positives: {false_positives}")
    print(f"  - False Negatives: {false_negatives}")
    print(f"  - True Negatives: {true_negatives}")

    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"  - Precision: {precision:.4f}")
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"  - Recall: {recall:.4f}")

    return detector


def demonstrate_zk_primitives():
    """Demonstrate zero-knowledge cryptographic primitives"""
    print("\n" + "="*60)
    print("Zero-Knowledge Primitives Demo")
    print("="*60)

    # Create ZK primitives
    zk = create_zk_primitives()

    # 1. Commitment scheme
    print("\n1. Pedersen Commitment:")
    secret_value = 42
    commitment = zk.generate_commitment(secret_value)
    print(f"  - Secret value: {secret_value}")
    print(f"  - Commitment: {commitment.value[:32]}...")
    print(f"  - Hiding: {commitment.hiding}")
    print(f"  - Binding: {commitment.binding}")

    # 2. Schnorr proof
    print("\n2. Schnorr Zero-Knowledge Proof:")
    secret_key = 12345
    message = "I know the secret key"
    proof = zk.schnorr_prove(secret_key, message)
    print(f"  - Message: {message}")
    print(f"  - Proof type: {proof.proof_type.value}")
    print(f"  - Commitment: {proof.commitment[:32]}...")
    print(f"  - Challenge: {proof.challenge[:32]}...")
    print(f"  - Response: {proof.response[:32]}...")

    # Verify proof
    is_valid = zk.schnorr_verify(proof)
    print(f"  - Proof verified: {is_valid}")

    # 3. Range proof
    print("\n3. Range Proof (proving 0 <= value < 2^32):")
    value = 1000
    range_proof = zk.range_proof(value, bits=32)
    print(f"  - Value: {value}")
    print(f"  - Bits: {range_proof['bits']}")
    print(f"  - Bit commitments: {len(range_proof['bit_commitments'])}")
    print(f"  - Bit proofs: {len(range_proof['bit_proofs'])}")

    # 4. Set membership
    print("\n4. Set Membership Proof:")
    from zkaedi.core.zk_primitives import ZKAccumulator
    accumulator = ZKAccumulator(zk)

    # Add elements to set
    elements = ["alice@example.com", "bob@example.com", "charlie@example.com"]
    for elem in elements:
        accumulator.add(elem)

    # Prove membership
    member = "bob@example.com"
    membership_proof = accumulator.prove_membership(member)
    print(f"  - Set size: {len(elements)}")
    print(f"  - Proving membership of: {member}")
    print(f"  - Proof generated: {membership_proof is not None}")

    return zk


def demonstrate_authenticated_encryption():
    """Demonstrate authenticated encryption with ZK properties"""
    print("\n" + "="*60)
    print("Authenticated Encryption Demo")
    print("="*60)

    # Create encryption instance
    encryption = create_zk_authenticated_encryption()

    # Sensitive data
    sensitive_data = {
        "patient_id": "P123456",
        "diagnosis": "Type 2 Diabetes",
        "treatment": "Metformin 500mg",
        "risk_score": 0.73
    }

    print("\n1. Encrypting sensitive data:")
    print(f"  - Original data: {sensitive_data}")

    # Encrypt with proof
    proof_data = {
        "data_type": "medical_record",
        "classification": "confidential",
        "authorized_roles": ["doctor", "nurse"]
    }

    encrypted, commitment = encryption.encrypt_with_proof(
        json.dumps(sensitive_data),
        proof_data
    )

    print(f"\n  - Encrypted ciphertext: {encrypted.ciphertext[:32]}...")
    print(f"  - Nonce: {encrypted.nonce[:16]}...")
    print(f"  - Authentication tag: {encrypted.tag[:16]}...")
    print(f"  - Data commitment: {commitment[:32]}...")

    # Create data proof without revealing content
    print("\n2. Creating zero-knowledge proof about encrypted data:")
    data_proof = encryption.create_data_proof(
        json.dumps(sensitive_data).encode(),
        "This medical record contains diabetes diagnosis"
    )

    print(f"  - Statement: {data_proof['statement']}")
    print(f"  - Proof type: {data_proof['proof_type']}")
    print(f"  - Timestamp: {data_proof['timestamp']}")
    print(f"  - Signature: {data_proof['signature'][:32]}...")

    # Verify proof
    is_valid = encryption.verify_data_proof(data_proof)
    print(f"  - Proof verified: {is_valid}")

    # Decrypt with verification
    print("\n3. Decrypting with commitment verification:")
    decrypted_bytes, verified = encryption.decrypt_with_verification(
        encrypted,
        commitment
    )

    if verified:
        decrypted_data = json.loads(decrypted_bytes.decode('utf-8'))
        print(f"  - Decryption successful: {verified}")
        print(f"  - Recovered data: {decrypted_data}")
    else:
        print("  - Decryption failed: commitment mismatch")

    return encryption


def main():
    """Main demonstration function"""
    print("\n" + "="*60)
    print("zkAEDI - Zero-Knowledge Authenticated Encrypted Data Intelligence")
    print("Master-Tier Implementation by iDeaKz")
    print("="*60)

    # 1. Generate datasets
    datasets = demonstrate_dataset_generation()

    # 2. Train models with ZK proofs
    model = demonstrate_model_training(datasets)

    # 3. Anomaly detection
    detector = demonstrate_anomaly_detection(datasets)

    # 4. ZK primitives
    zk_primitives = demonstrate_zk_primitives()

    # 5. Authenticated encryption
    encryption = demonstrate_authenticated_encryption()

    print("\n" + "="*60)
    print("zkAEDI Demonstration Complete!")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Synthetic dataset generation (financial, healthcare, IoT, cybersecurity)")
    print("  ✓ Privacy-preserving model training with differential privacy")
    print("  ✓ Zero-knowledge proofs for model integrity")
    print("  ✓ Anomaly detection with cryptographic guarantees")
    print("  ✓ Authenticated encryption with ZK properties")
    print("  ✓ Comprehensive privacy and security reporting")
    print("\nAll components working together for maximum security and intelligence!")


if __name__ == "__main__":
    import json  # Import at module level for the encryption demo
    main()
