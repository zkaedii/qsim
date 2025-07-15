const exportABI = withErrorHandling( function ()
{
    const abi = [
        {
            "inputs": [ { "internalType": "address", "name": "initialOwner", "type": "address" } ],
            "stateMutability": "nonpayable",
            "type": "constructor"
        },
        {
            "inputs": [ { "internalType": "address", "name": "to", "type": "address" }, { "internalType": "uint256", "name": "amount", "type": "uint256" } ],
            "name": "mint",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [ { "internalType": "uint256", "name": "amount", "type": "uint256" }, { "internalType": "uint256", "name": "duration", "type": "uint256" } ],
            "name": "stake",
            "outputs": [ { "internalType": "uint256", "name": "", "type": "uint256" } ],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "totalSupply",
            "outputs": [ { "internalType": "uint256", "name": "", "type": "uint256" } ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [ { "internalType": "uint256", "name": "stakeIndex", "type": "uint256" } ],
            "name": "unstake",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [ { "internalType": "string", "name": "title", "type": "string" }, { "internalType": "string", "name": "description", "type": "string" } ],
            "name": "createProposal",
            "outputs": [ { "internalType": "bytes32", "name": "", "type": "bytes32" } ],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [ { "internalType": "bytes32", "name": "proposalId", "type": "bytes32" }, { "internalType": "uint8", "name": "choice", "type": "uint8" } ],
            "name": "castVote",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [ { "internalType": "uint256", "name": "modelId", "type": "uint256" }, { "internalType": "uint8", "name": "trainingType", "type": "uint8" } ],
            "name": "startTrainingSession",
            "outputs": [ { "internalType": "uint256", "name": "", "type": "uint256" } ],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [ { "internalType": "address", "name": "poolAddress", "type": "address" }, { "internalType": "uint256", "name": "amount", "type": "uint256" } ],
            "name": "stakeLiquidity",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [ { "internalType": "uint256", "name": "targetChain", "type": "uint256" }, { "internalType": "address", "name": "recipient", "type": "address" }, { "internalType": "uint256", "name": "amount", "type": "uint256" } ],
            "name": "bridgeTransfer",
            "outputs": [ { "internalType": "bytes32", "name": "", "type": "bytes32" } ],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "executeBuybackAndBurn",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [ { "internalType": "string", "name": "reason", "type": "string" } ],
            "name": "activateEmergency",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [ { "internalType": "address", "name": "user", "type": "address" } ],
            "name": "getUserStakes",
            "outputs": [ { "internalType": "tuple[]", "name": "", "type": "tuple[]" } ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "getContractStats",
            "outputs": [ { "internalType": "uint256", "name": "totalSupplyAmount", "type": "uint256" }, { "internalType": "uint256", "name": "totalStakedAmount", "type": "uint256" }, { "internalType": "uint256", "name": "totalRewardsAmount", "type": "uint256" } ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "anonymous": false,
            "inputs": [ { "indexed": true, "internalType": "address", "name": "user", "type": "address" }, { "indexed": true, "internalType": "uint256", "name": "stakeIndex", "type": "uint256" }, { "indexed": false, "internalType": "uint256", "name": "amount", "type": "uint256" } ],
            "name": "Staked",
            "type": "event"
        },
        {
            "anonymous": false,
            "inputs": [ { "indexed": true, "internalType": "bytes32", "name": "proposalId", "type": "bytes32" }, { "indexed": true, "internalType": "address", "name": "proposer", "type": "address" } ],
            "name": "ProposalCreated",
            "type": "event"
        },
        {
            "anonymous": false,
            "inputs": [ { "indexed": true, "internalType": "uint256", "name": "sessionId", "type": "uint256" }, { "indexed": true, "internalType": "address", "name": "trainer", "type": "address" } ],
            "name": "AITrainingStarted",
            "type": "event"
        }
    ];

    const abiJson = JSON.stringify( abi, null, 2 );
    const blob = new Blob( [ abiJson ], { type: 'application/json' } );
    const url = URL.createObjectURL( blob );

    const a = document.createElement( 'a' );
    a.href = url;
    a.download = `HModelToken_ABI_${ new Date().toISOString().split( 'T' )[ 0 ] }.json`;
    a.style.display = 'none';
    document.body.appendChild( a );
    a.click();
    document.body.removeChild( a );
    URL.revokeObjectURL( url );

    // Also copy to clipboard
    navigator.clipboard.writeText( abiJson ).then( () =>
    {
        showToast( 'ABI exported and copied to clipboard!', 'success' );
    } ).catch( () =>
    {
        showToast( 'ABI exported to file successfully!', 'success' );
    } );

    logOperation( 'Contract ABI exported successfully', 'success' );
}, 'ABI Export' );

// ==================== TOKEN MANAGEMENT FUNCTIONS ====================
const mintTokens = withErrorHandling( async function ()
{
    if ( !SystemState.isDeployed )
    {
        throw new Error( 'Contract must be deployed first' );
    }

    const recipient = prompt( 'Enter recipient address:' );
    const amount = prompt( 'Enter amount to mint (in tokens):' );

    if ( !recipient || !amount )
    {
        throw new Error( 'Recipient and amount are required' );
    }

    SecurityManager.validateInput( recipient, 'string', 100 );

    // Simulate minting process
    showToast( 'Minting tokens...', 'info' );
    await new Promise( resolve => setTimeout( resolve, 2000 ) );

    const transactionHash = '0x' + Array.from( { length: 64 }, () => Math.floor( Math.random() * 16 ).toString( 16 ) ).join( '' );
    document.getElementById( 'transaction-hash' ).value = transactionHash;

    // Update metrics
    SystemState.metrics.totalSupply += parseInt( amount );
    updateTokenMetrics();

    showToast( `Successfully minted ${ amount } tokens to ${ recipient.slice( 0, 10 ) }...`, 'success' );
    logOperation( `Minted ${ amount } tokens to ${ recipient }`, 'success' );
}, 'Token Minting' );

const burnTokens = withErrorHandling( async function ()
{
    if ( !SystemState.isDeployed )
    {
        throw new Error( 'Contract must be deployed first' );
    }

    const amount = prompt( 'Enter amount to burn (in tokens):' );
    if ( !amount )
    {
        throw new Error( 'Amount is required' );
    }

    const confirmation = confirm( `Are you sure you want to burn ${ amount } tokens? This action cannot be undone.` );
    if ( !confirmation )
    {
        throw new Error( 'Burn operation cancelled by user' );
    }

    // Simulate burning process
    showToast( 'Burning tokens...', 'info' );
    await new Promise( resolve => setTimeout( resolve, 2000 ) );

    const transactionHash = '0x' + Array.from( { length: 64 }, () => Math.floor( Math.random() * 16 ).toString( 16 ) ).join( '' );
    document.getElementById( 'transaction-hash' ).value = transactionHash;

    // Update metrics
    SystemState.metrics.totalSupply -= parseInt( amount );
    updateTokenMetrics();

    showToast( `Successfully burned ${ amount } tokens`, 'success' );
    logOperation( `Burned ${ amount } tokens`, 'success' );
}, 'Token Burning' );

const distributeTokens = withErrorHandling( async function ()
{
    if ( !SystemState.isDeployed )
    {
        throw new Error( 'Contract must be deployed first' );
    }

    const distributionData = prompt( 'Enter distribution data (format: address1:amount1,address2:amount2):' );
    if ( !distributionData )
    {
        throw new Error( 'Distribution data is required' );
    }

    SecurityManager.validateInput( distributionData, 'string', 10000 );

    // Parse distribution data
    const distributions = distributionData.split( ',' ).map( item =>
    {
        const [ address, amount ] = item.split( ':' );
        return { address: address.trim(), amount: parseInt( amount.trim() ) };
    } );

    if ( distributions.some( d => !d.address || !d.amount ) )
    {
        throw new Error( 'Invalid distribution format' );
    }

    // Simulate batch distribution
    showToast( 'Distributing tokens to multiple addresses...', 'info' );

    for ( let i = 0; i < distributions.length; i++ )
    {
        await new Promise( resolve => setTimeout( resolve, 1000 ) );
        showToast( `Distributing to address ${ i + 1 }/${ distributions.length }...`, 'info' );
    }

    const totalDistributed = distributions.reduce( ( sum, d ) => sum + d.amount, 0 );
    SystemState.metrics.holders += distributions.length;
    updateTokenMetrics();

    showToast( `Successfully distributed ${ totalDistributed } tokens to ${ distributions.length } addresses`, 'success' );
    logOperation( `Batch distribution completed: ${ totalDistributed } tokens to ${ distributions.length } recipients`, 'success' );
}, 'Token Distribution' );

// ==================== AI INTELLIGENCE FUNCTIONS ====================
const startAITraining = withErrorHandling( async function ()
{
    const modelType = document.getElementById( 'ai-model-type' ).value;
    const trainingData = document.getElementById( 'training-data' ).value;
    const accuracyTarget = document.getElementById( 'accuracy-target' ).value;

    if ( !trainingData.trim() )
    {
        throw new Error( 'Training data is required' );
    }

    SecurityManager.validateInput( trainingData, 'string', 50000 );

    const sessionId = Math.floor( Math.random() * 1000000 );
    const trainingSession = {
        id: sessionId,
        modelType: modelType,
        startTime: new Date().toISOString(),
        targetAccuracy: parseInt( accuracyTarget ),
        status: 'training',
        progress: 0,
        epochs: 0,
        currentAccuracy: 0
    };

    SystemState.aiModels.set( sessionId, trainingSession );

    const resultsDiv = document.getElementById( 'ai-results' );
    const contentDiv = document.getElementById( 'ai-content' );

    resultsDiv.style.display = 'block';
    resultsDiv.classList.add( 'show' );

    // Simulate training process
    const trainingSteps = [
        'Initializing neural network...',
        'Loading training dataset...',
        'Preprocessing data...',
        'Starting training epochs...',
        'Optimizing hyperparameters...',
        'Validating model performance...',
        'Fine-tuning weights...',
        'Training completed!'
    ];

    for ( let i = 0; i < trainingSteps.length; i++ )
    {
        const progress = ( ( i + 1 ) / trainingSteps.length ) * 100;
        const epochs = Math.floor( ( i + 1 ) * 10 );
        const accuracy = Math.min( parseInt( accuracyTarget ), 50 + ( progress / 100 ) * 45 + Math.random() * 5 );

        trainingSession.progress = progress;
        trainingSession.epochs = epochs;
        trainingSession.currentAccuracy = accuracy;

        contentDiv.innerHTML = `
                    <div style="background: var(--dark-gradient); padding: 20px; border-radius: 12px; color: white;">
                        <h4>🧠 AI Training Session #${ sessionId }</h4>
                        <p><strong>Model Type:</strong> ${ modelType.replace( '_', ' ' ).toUpperCase() }</p>
                        <p><strong>Current Step:</strong> ${ trainingSteps[ i ] }</p>
                        <p><strong>Progress:</strong> ${ progress.toFixed( 1 ) }%</p>
                        <p><strong>Epochs:</strong> ${ epochs }/100</p>
                        <p><strong>Current Accuracy:</strong> ${ accuracy.toFixed( 2 ) }%</p>
                        <p><strong>Target Accuracy:</strong> ${ accuracyTarget }%</p>
                        
                        <div class="progress-container">
                            <div class="progress-label">
                                <span>Training Progress</span>
                                <span>${ progress.toFixed( 1 ) }%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${ progress }%;"></div>
                            </div>
                        </div>
                        
                        <div class="progress-container">
                            <div class="progress-label">
                                <span>Accuracy Progress</span>
                                <span>${ accuracy.toFixed( 1 ) }%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${ ( accuracy / parseInt( accuracyTarget ) ) * 100 }%;"></div>
                            </div>
                        </div>
                    </div>
                `;

        await new Promise( resolve => setTimeout( resolve, 2000 ) );
    }

    trainingSession.status = 'completed';
    trainingSession.endTime = new Date().toISOString();

    // Final results
    const finalAccuracy = Math.min( parseInt( accuracyTarget ), 85 + Math.random() * 10 );
    const modelSize = Math.floor( Math.random() * 500 + 100 ); // MB
    const inferenceTime = ( Math.random() * 50 + 10 ).toFixed( 2 ); // ms

    contentDiv.innerHTML += `
                <div style="background: var(--success-gradient); padding: 20px; border-radius: 12px; color: white; margin-top: 15px;">
                    <h4>🎉 Training Completed Successfully!</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; font-weight: bold;">${ finalAccuracy.toFixed( 2 ) }%</div>
                            <div style="opacity: 0.9;">Final Accuracy</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; font-weight: bold;">${ modelSize }MB</div>
                            <div style="opacity: 0.9;">Model Size</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; font-weight: bold;">${ inferenceTime }ms</div>
                            <div style="opacity: 0.9;">Inference Time</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; font-weight: bold;">A+</div>
                            <div style="opacity: 0.9;">Model Grade</div>
                        </div>
                    </div>
                    <p style="margin-top: 15px; text-align: center; font-weight: bold;">
                        🏆 Model ready for deployment and token rewards!
                    </p>
                </div>
            `;

    logOperation( `AI training completed: ${ modelType } model achieved ${ finalAccuracy.toFixed( 2 ) }% accuracy`, 'success' );
}, 'AI Training' );

const deployAIModel = withErrorHandling( async function ()
{
    const activeSessions = Array.from( SystemState.aiModels.values() ).filter( s => s.status === 'completed' );

    if ( activeSessions.length === 0 )
    {
        throw new Error( 'No trained models available for deployment' );
    }

    const latestSession = activeSessions[ activeSessions.length - 1 ];

    showToast( 'Deploying AI model to blockchain...', 'info' );

    // Simulate deployment process
    const deploymentSteps = [
        'Serializing model weights...',
        'Compressing model data...',
        'Uploading to IPFS...',
        'Creating model NFT...',
        'Recording on blockchain...',
        'Deployment complete!'
    ];

    for ( let i = 0; i < deploymentSteps.length; i++ )
    {
        showToast( deploymentSteps[ i ], 'info' );
        await new Promise( resolve => setTimeout( resolve, 1500 ) );
    }

    const modelHash = 'Qm' + Array.from( { length: 44 }, () => Math.random().toString( 36 ).charAt( 0 ) ).join( '' );
    const nftTokenId = Math.floor( Math.random() * 10000 );

    const resultsDiv = document.getElementById( 'ai-results' );
    const contentDiv = document.getElementById( 'ai-content' );

    contentDiv.innerHTML += `
                <div style="background: var(--primary-gradient); padding: 20px; border-radius: 12px; color: white; margin-top: 15px;">
                    <h4>🚀 Model Deployed Successfully!</h4>
                    <p><strong>Model ID:</strong> ${ latestSession.id }</p>
                    <p><strong>IPFS Hash:</strong> <code style="background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px;">${ modelHash }</code></p>
                    <p><strong>NFT Token ID:</strong> #${ nftTokenId }</p>
                    <p><strong>Deployment Network:</strong> ${ SystemState.currentNetwork }</p>
                    <p><strong>Model Accuracy:</strong> ${ latestSession.currentAccuracy.toFixed( 2 ) }%</p>
                    <p><strong>Access URL:</strong> <code style="background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px;">https://api.hmodel.ai/models/${ latestSession.id }</code></p>
                    <p style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px;">
                        💡 <strong>Pro Tip:</strong> Your deployed model can now earn HModel tokens through inference usage and accuracy rewards!
                    </p>
                </div>
            `;

    resultsDiv.style.display = 'block';

    logOperation( `AI model deployed: ID ${ latestSession.id }, Hash ${ modelHash }`, 'success' );
    showToast( 'AI model deployed and earning tokens!', 'success' );
}, 'AI Model Deployment' );

const generateEmbeddings = withErrorHandling( async function ()
{
    const inputText = document.getElementById( 'training-data' ).value;

    if ( !inputText.trim() )
    {
        throw new Error( 'Input text is required for embedding generation' );
    }

    SecurityManager.validateInput( inputText, 'string', 10000 );

    showToast( 'Generating vector embeddings...', 'info' );

    // Simulate embedding generation
    const embeddingSteps = [
        'Tokenizing input text...',
        'Loading transformer model...',
        'Computing attention weights...',
        'Generating embeddings...',
        'Normalizing vectors...',
        'Embeddings ready!'
    ];

    for ( let i = 0; i < embeddingSteps.length; i++ )
    {
        showToast( embeddingSteps[ i ], 'info' );
        await new Promise( resolve => setTimeout( resolve, 1000 ) );
    }

    // Generate synthetic embedding data
    const dimension = 512;
    const embedding = Array.from( { length: dimension }, () => ( Math.random() - 0.5 ) * 2 );
    const magnitude = Math.sqrt( embedding.reduce( ( sum, val ) => sum + val * val, 0 ) );
    const normalizedEmbedding = embedding.map( val => val / magnitude );

    const embeddingId = SecurityManager.generateSecureToken().slice( 0, 16 );
    SystemState.embeddings.set( embeddingId, {
        id: embeddingId,
        text: inputText.slice( 0, 100 ) + '...',
        embedding: normalizedEmbedding,
        dimension: dimension,
        created: new Date().toISOString(),
        similarity: {}
    } );

    const resultsDiv = document.getElementById( 'ai-results' );
    const contentDiv = document.getElementById( 'ai-content' );

    contentDiv.innerHTML += `
                <div style="background: var(--secondary-gradient); padding: 20px; border-radius: 12px; color: white; margin-top: 15px;">
                    <h4>🧬 Vector Embedding Generated!</h4>
                    <p><strong>Embedding ID:</strong> <code style="background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px;">${ embeddingId }</code></p>
                    <p><strong>Dimension:</strong> ${ dimension }D</p>
                    <p><strong>Input Length:</strong> ${ inputText.length } characters</p>
                    <p><strong>Magnitude:</strong> ${ magnitude.toFixed( 6 ) }</p>
                    <p><strong>Sample Values:</strong> [${ normalizedEmbedding.slice( 0, 5 ).map( v => v.toFixed( 4 ) ).join( ', ' ) }...]</p>
                    
                    <div style="margin-top: 15px;">
                        <p><strong>🎯 Quality Metrics:</strong></p>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px;">
                            <div style="text-align: center; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px;">
                                <div style="font-size: 1.5rem; font-weight: bold;">98.7%</div>
                                <div style="font-size: 0.9rem;">Coherence</div>
                            </div>
                            <div style="text-align: center; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px;">
                                <div style="font-size: 1.5rem; font-weight: bold;">96.2%</div>
                                <div style="font-size: 0.9rem;">Semantic</div>
                            </div>
                            <div style="text-align: center; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px;">
                                <div style="font-size: 1.5rem; font-weight: bold;">94.8%</div>
                                <div style="font-size: 0.9rem;">Uniqueness</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

    logOperation( `Vector embedding generated: ID ${ embeddingId }, dimension ${ dimension }`, 'success' );
    showToast( 'Vector embedding generated successfully!', 'success' );
}, 'Vector Embedding Generation' );

// ==================== VECTOR EMBEDDING GENIUS FUNCTIONS ====================
const generateVectorEmbedding = withErrorHandling( async function ()
{
    const inputData = document.getElementById( 'embedding-input' ).value;
    const dimension = parseInt( document.getElementById( 'embedding-dimension' ).value );
    const method = document.getElementById( 'embedding-method' ).value;

    if ( !inputData.trim() )
    {
        throw new Error( 'Input data is required for embedding generation' );
    }

    SecurityManager.validateInput( inputData, 'string', 50000 );

    const resultsDiv = document.getElementById( 'embedding-results' );
    const contentDiv = document.getElementById( 'embedding-content' );

    resultsDiv.style.display = 'block';
    resultsDiv.classList.add( 'show' );

    // Simulate advanced embedding generation
    const embeddingSteps = [
        'Analyzing input structure...',
        'Initializing embedding model...',
        'Computing contextual representations...',
        'Applying dimensionality reduction...',
        'Optimizing vector space...',
        'Finalizing embeddings...'
    ];

    for ( let i = 0; i < embeddingSteps.length; i++ )
    {
        const progress = ( ( i + 1 ) / embeddingSteps.length ) * 100;

        contentDiv.innerHTML = `
                    <div style="background: var(--dark-gradient); padding: 20px; border-radius: 12px; color: white;">
                        <h4>🧬 Vector Embedding Generation</h4>
                        <p><strong>Method:</strong> ${ method.replace( '_', ' ' ).toUpperCase() }</p>
                        <p><strong>Dimension:</strong> ${ dimension }D</p>
                        <p><strong>Current Step:</strong> ${ embeddingSteps[ i ] }</p>
                        
                        <div class="progress-container">
                            <div class="progress-label">
                                <span>Generation Progress</span>
                                <span>${ progress.toFixed( 1 ) }%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${ progress }%;"></div>
                            </div>
                        </div>
                    </div>
                `;

        await new Promise( resolve => setTimeout( resolve, 1500 ) );
    }

    // Generate high-quality synthetic embedding
    const embedding = generateAdvancedEmbedding( inputData, dimension, method );
    const embeddingId = SecurityManager.generateSecureToken().slice( 0, 12 );

    // Calculate advanced metrics
    const coherenceScore = 92 + Math.random() * 6;
    const semanticScore = 88 + Math.random() * 8;
    const uniquenessScore = 85 + Math.random() * 10;
    const qualityGrade = coherenceScore > 95 ? 'A+' : coherenceScore > 90 ? 'A' : coherenceScore > 85 ? 'B+' : 'B';

    SystemState.embeddings.set( embeddingId, {
        id: embeddingId,
        data: inputData,
        embedding: embedding,
        dimension: dimension,
        method: method,
        metrics: { coherenceScore, semanticScore, uniquenessScore },
        created: new Date().toISOString()
    } );

    contentDiv.innerHTML = `
                <div style="background: var(--success-gradient); padding: 25px; border-radius: 12px; color: white;">
                    <h4>🎉 Vector Embedding Generated Successfully!</h4>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
                        <div style="text-align: center; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;">
                            <div style="font-size: 1.8rem; font-weight: bold;">${ dimension }D</div>
                            <div style="opacity: 0.9;">Dimension</div>
                        </div>
                        <div style="text-align: center; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;">
                            <div style="font-size: 1.8rem; font-weight: bold;">${ qualityGrade }</div>
                            <div style="opacity: 0.9;">Quality Grade</div>
                        </div>
                        <div style="text-align: center; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;">
                            <div style="font-size: 1.8rem; font-weight: bold;">${ embedding.magnitude.toFixed( 3 ) }</div>
                            <div style="opacity: 0.9;">Magnitude</div>
                        </div>
                        <div style="text-align: center; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;">
                            <div style="font-size: 1.8rem; font-weight: bold;">${ method.split( '_' )[ 0 ] }</div>
                            <div style="opacity: 0.9;">Method</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <h5>📊 Quality Analysis</h5>
                        <div style="margin-top: 10px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span>Coherence Score</span>
                                <span>${ coherenceScore.toFixed( 1 ) }%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${ coherenceScore }%; background: linear-gradient(90deg, #4ade80, #06b6d4);"></div>
                            </div>
                        </div>
                        <div style="margin-top: 10px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span>Semantic Quality</span>
                                <span>${ semanticScore.toFixed( 1 ) }%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${ semanticScore }%; background: linear-gradient(90deg, #f59e0b, #f97316);"></div>
                            </div>
                        </div>
                        <div style="margin-top: 10px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span>Uniqueness</span>
                                <span>${ uniquenessScore.toFixed( 1 ) }%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${ uniquenessScore }%; background: linear-gradient(90deg, #8b5cf6, #a855f7);"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                        <p><strong>Embedding ID:</strong> <code>${ embeddingId }</code></p>
                        <p><strong>Vector Preview:</strong> [${ embedding.vector.slice( 0, 5 ).map( v => v.toFixed( 4 ) ).join( ', ' ) }...]</p>
                        <p><strong>Similarity Ready:</strong> ✅ Available for comparison</p>
                    </div>
                </div>
            `;

    // Update chart
    updateEmbeddingChart( embedding );

    logOperation( `Vector embedding generated: ${ embeddingId } (${ dimension }D, ${ method })`, 'success' );
}, 'Vector Embedding Generation' );

const compareEmbeddings = withErrorHandling( async function ()
{
    const embeddings = Array.from( SystemState.embeddings.values() );

    if ( embeddings.length < 2 )
    {
        throw new Error( 'At least 2 embeddings are required for comparison' );
    }

    showToast( 'Computing embedding similarities...', 'info' );

    // Compute all pairwise similarities
    const similarities = [];
    for ( let i = 0; i < embeddings.length; i++ )
    {
        for ( let j = i + 1; j < embeddings.length; j++ )
        {
            const sim = computeCosineSimilarity( embeddings[ i ].embedding.vector, embeddings[ j ].embedding.vector );
            similarities.push( {
                embedding1: embeddings[ i ],
                embedding2: embeddings[ j ],
                similarity: sim,
                distance: 1 - sim
            } );
        }
    }

    // Sort by similarity
    similarities.sort( ( a, b ) => b.similarity - a.similarity );

    const resultsDiv = document.getElementById( 'embedding-results' );
    const contentDiv = document.getElementById( 'embedding-content' );

    contentDiv.innerHTML += `
                <div style="background: var(--primary-gradient); padding: 20px; border-radius: 12px; color: white; margin-top: 15px;">
                    <h4>📊 Embedding Similarity Analysis</h4>
                    <p><strong>Total Embeddings:</strong> ${ embeddings.length }</p>
                    <p><strong>Comparisons Computed:</strong> ${ similarities.length }</p>
                    
                    <div style="margin-top: 20px;">
                        <h5>🏆 Top Similarities</h5>
                        ${ similarities.slice( 0, 5 ).map( ( sim, index ) => `
                            <div style="background: rgba(0,0,0,0.2); padding: 12px; margin: 8px 0; border-radius: 8px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div style="flex: 1;">
                                        <div style="font-weight: bold;">#${ index + 1 } - ${ ( sim.similarity * 100 ).toFixed( 2 ) }% Similar</div>
                                        <div style="font-size: 0.9rem; opacity: 0.8;">
                                            ${ sim.embedding1.id } ↔ ${ sim.embedding2.id }
                                        </div>
                                    </div>
                                    <div style="width: 100px;">
                                        <div class="progress-bar">
                                            <div class="progress-fill" style="width: ${ sim.similarity * 100 }%;"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `).join( '' ) }
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <h5>📈 Statistical Summary</h5>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-top: 10px;">
                            <div style="text-align: center; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px;">
                                <div style="font-size: 1.3rem; font-weight: bold;">${ ( Math.max( ...similarities.map( s => s.similarity ) ) * 100 ).toFixed( 1 ) }%</div>
                                <div style="font-size: 0.8rem;">Max Similarity</div>
                            </div>
                            <div style="text-align: center; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px;">
                                <div style="font-size: 1.3rem; font-weight: bold;">${ ( similarities.reduce( ( sum, s ) => sum + s.similarity, 0 ) / similarities.length * 100 ).toFixed( 1 ) }%</div>
                                <div style="font-size: 0.8rem;">Avg Similarity</div>
                            </div>
                            <div style="text-align: center; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px;">
                                <div style="font-size: 1.3rem; font-weight: bold;">${ ( Math.min( ...similarities.map( s => s.similarity ) ) * 100 ).toFixed( 1 ) }%</div>
                                <div style="font-size: 0.8rem;">Min Similarity</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

    logOperation( `Embedding comparison completed: ${ similarities.length } pairwise comparisons`, 'success' );
}, 'Embedding Comparison' );

const optimizeEmbedding = withErrorHandling( async function ()
{
    const embeddings = Array.from( SystemState.embeddings.values() );

    if ( embeddings.length === 0 )
    {
        throw new Error( 'No embeddings available for optimization' );
    }

    const latestEmbedding = embeddings[ embeddings.length - 1 ];

    showToast( 'Optimizing embedding quality...', 'info' );

    // Simulate optimization process
    const optimizationSteps = [
        'Analyzing vector distribution...',
        'Identifying optimization targets...',
        'Applying PCA transformation...',
        'Normalizing vector space...',
        'Fine-tuning dimensions...',
        'Validation and quality check...',
        'Optimization complete!'
    ];

    const resultsDiv = document.getElementById( 'embedding-results' );
    const contentDiv = document.getElementById( 'embedding-content' );

    for ( let i = 0; i < optimizationSteps.length; i++ )
    {
        const progress = ( ( i + 1 ) / optimizationSteps.length ) * 100;
        showToast( optimizationSteps[ i ], 'info' );
        await new Promise( resolve => setTimeout( resolve, 1000 ) );
    }

    // Generate optimized embedding
    const optimizedEmbedding = generateAdvancedEmbedding(
        latestEmbedding.data,
        latestEmbedding.dimension,
        'optimized_' + latestEmbedding.method
    );

    const improvementScore = 5 + Math.random() * 10;
    const newCoherence = Math.min( 98, latestEmbedding.metrics.coherenceScore + improvementScore );
    const newSemantic = Math.min( 98, latestEmbedding.metrics.semanticScore + improvementScore );
    const newUniqueness = Math.min( 98, latestEmbedding.metrics.uniquenessScore + improvementScore );

    // Update the embedding
    latestEmbedding.embedding = optimizedEmbedding;
    latestEmbedding.metrics.coherenceScore = newCoherence;
    latestEmbedding.metrics.semanticScore = newSemantic;
    latestEmbedding.metrics.uniquenessScore = newUniqueness;
    latestEmbedding.optimized = true;

    contentDiv.innerHTML += `
                <div style="background: var(--warning-gradient); padding: 20px; border-radius: 12px; color: white; margin-top: 15px;">
                    <h4>⚡ Embedding Optimization Complete!</h4>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                        <div>
                            <h5>📈 Performance Improvements</h5>
                            <div style="margin-top: 10px;">
                                <p><strong>Coherence:</strong> +${ improvementScore.toFixed( 1 ) }% → ${ newCoherence.toFixed( 1 ) }%</p>
                                <p><strong>Semantic Quality:</strong> +${ improvementScore.toFixed( 1 ) }% → ${ newSemantic.toFixed( 1 ) }%</p>
                                <p><strong>Uniqueness:</strong> +${ improvementScore.toFixed( 1 ) }% → ${ newUniqueness.toFixed( 1 ) }%</p>
                            </div>
                        </div>
                        <div>
                            <h5>🔧 Optimization Details</h5>
                            <div style="margin-top: 10px;">
                                <p><strong>Vector Magnitude:</strong> ${ optimizedEmbedding.magnitude.toFixed( 6 ) }</p>
                                <p><strong>Sparsity Reduced:</strong> ${ ( Math.random() * 15 + 5 ).toFixed( 1 ) }%</p>
                                <p><strong>Quality Grade:</strong> A+ 🏆</p>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px; text-align: center;">
                        <p style="font-size: 1.1rem; font-weight: bold;">
                            🎯 Embedding optimized and ready for production use!
                        </p>
                        <p style="margin-top: 5px; opacity: 0.9;">
                            Enhanced performance for similarity search and ML applications
                        </p>
                    </div>
                </div>
            `;

    logOperation( `Embedding optimized: ID ${ latestEmbedding.id }, improvement +${ improvementScore.toFixed( 1 ) }%`, 'success' );
}, 'Embedding Optimization' );

// ==================== BLOCKCHAIN INTEGRATION FUNCTIONS ====================
const connectWallet = withErrorHandling( async function ()
{
    if ( typeof window.ethereum === 'undefined' )
    {
        throw new Error( 'MetaMask or Web3 wallet not detected. Please install a Web3 wallet.' );
    }

    showToast( 'Connecting to Web3 wallet...', 'info' );

    try
    {
        // Request account access
        const accounts = await window.ethereum.request( { method: 'eth_requestAccounts' } );

        if ( accounts.length === 0 )
        {
            throw new Error( 'No accounts found. Please unlock your wallet.' );
        }

        SystemState.walletConnected = true;
        const userAccount = accounts[ 0 ];

        // Initialize Web3
        if ( typeof Web3 !== 'undefined' )
        {
            SystemState.web3Instance = new Web3( window.ethereum );

            // Get network info
            const networkId = await SystemState.web3Instance.eth.net.getId();
            const balance = await SystemState.web3Instance.eth.getBalance( userAccount );
            const balanceEth = SystemState.web3Instance.utils.fromWei( balance, 'ether' );

            // Update UI
            document.getElementById( 'smart-contract' ).value = userAccount;

            showToast( `Wallet connected: ${ userAccount.slice( 0, 10 ) }...`, 'success' );
            logOperation( `Wallet connected: ${ userAccount }, Network: ${ networkId }, Balance: ${ parseFloat( balanceEth ).toFixed( 4 ) } ETH`, 'success' );

            // Update network status
            const networkNames = {
                1: 'Ethereum Mainnet',
                3: 'Ropsten Testnet',
                4: 'Rinkeby Testnet',
                137: 'Polygon Mainnet',
                80001: 'Polygon Mumbai'
            };

            const networkName = networkNames[ networkId ] || `Network ${ networkId }`;
            showToast( `Connected to ${ networkName }`, 'info' );

        } else
        {
            throw new Error( 'Web3 library not loaded' );
        }

    } catch ( error )
    {
        SystemState.walletConnected = false;
        throw error;
    }

}, 'Wallet Connection' );

const checkBalance = withErrorHandling( async function ()
{
    if ( !SystemState.walletConnected || !SystemState.web3Instance )
    {
        throw new Error( 'Wallet not connected. Please connect your wallet first.' );
    }

    showToast( 'Checking wallet balance...', 'info' );

    const accounts = await SystemState.web3Instance.eth.getAccounts();
    const userAccount = accounts[ 0 ];

    // Check ETH balance
    const ethBalance = await SystemState.web3Instance.eth.getBalance( userAccount );
    const ethBalanceFormatted = SystemState.web3Instance.utils.fromWei( ethBalance, 'ether' );

    // Check token balance (if contract is deployed)
    let tokenBalance = '0';
    if ( SystemState.isDeployed && SystemState.contractAddress )
    {
        // Simulate token balance check
        tokenBalance = ( Math.random() * 10000 + 1000 ).toFixed( 2 );
    }

    // Get gas price
    const gasPrice = await SystemState.web3Instance.eth.getGasPrice();
    const gasPriceGwei = SystemState.web3Instance.utils.fromWei( gasPrice, 'gwei' );

    // Update UI elements
    document.getElementById( 'gas-price-display' ).textContent = parseFloat( gasPriceGwei ).toFixed( 0 );

    showToast( `ETH Balance: ${ parseFloat( ethBalanceFormatted ).toFixed( 4 ) } ETH`, 'success' );

    if ( SystemState.isDeployed )
    {
        showToast( `HModel Token Balance: ${ tokenBalance } HMAI`, 'info' );
    }

    logOperation( `Balance check: ${ parseFloat( ethBalanceFormatted ).toFixed( 4 ) } ETH, ${ tokenBalance } HMAI tokens`, 'success' );

}, 'Balance Check' );

const bridgeTokens = withErrorHandling( async function ()
{
    if ( !SystemState.isDeployed )
    {
        throw new Error( 'Contract must be deployed before bridging tokens' );
    }

    if ( !SystemState.walletConnected )
    {
        throw new Error( 'Wallet must be connected for bridging operations' );
    }

    const targetChain = prompt( 'Enter target chain ID (1=Ethereum, 137=Polygon, 56=BSC):' );
    const amount = prompt( 'Enter amount to bridge:' );
    const recipient = prompt( 'Enter recipient address (or leave empty for same address):' );

    if ( !targetChain || !amount )
    {
        throw new Error( 'Target chain and amount are required' );
    }

    SecurityManager.validateInput( targetChain, 'string', 10 );
    SecurityManager.validateInput( amount, 'string', 20 );

    const supportedChains = { '1': 'Ethereum', '137': 'Polygon', '56': 'BSC', '43114': 'Avalanche' };

    if ( !supportedChains[ targetChain ] )
    {
        throw new Error( `Unsupported target chain: ${ targetChain }` );
    }

    showToast( `Initiating bridge to ${ supportedChains[ targetChain ] }...`, 'info' );

    // Simulate bridging process
    const bridgeSteps = [
        'Validating bridge request...',
        'Locking tokens on source chain...',
        'Generating bridge proof...',
        'Submitting to bridge contract...',
        'Waiting for confirmations...',
        'Minting tokens on target chain...',
        'Bridge complete!'
    ];

    for ( let i = 0; i < bridgeSteps.length; i++ )
    {
        showToast( bridgeSteps[ i ], 'info' );
        await new Promise( resolve => setTimeout( resolve, 2000 ) );
    }

    const bridgeTransactionId = SecurityManager.generateSecureToken();
    const estimatedTime = Math.floor( Math.random() * 10 + 5 ); // 5-15 minutes

    showToast( `Bridge successful! Transaction ID: ${ bridgeTransactionId.slice( 0, 16 ) }...`, 'success' );
    showToast( `Estimated arrival time: ${ estimatedTime } minutes`, 'info' );

    // Update transaction hash
    document.getElementById( 'transaction-hash' ).value = '0x' + bridgeTransactionId;

    logOperation( `Bridge initiated: ${ amount } tokens to chain ${ targetChain }, TX: ${ bridgeTransactionId }`, 'success' );

}, 'Token Bridging' );

// ==================== UTILITY FUNCTIONS ====================
function generateAdvancedEmbedding ( data, dimension, method )
{
    // Generate high-quality synthetic embedding based on input data and method
    const embedding = new Array( dimension );
    const dataHash = hashString( data );

    // Seed random number generator with data hash for reproducibility
    let seed = parseInt( dataHash.slice( 0, 8 ), 16 );

    for ( let i = 0; i < dimension; i++ )
    {
        // Generate pseudo-random values based on method
        seed = ( seed * 1103515245 + 12345 ) & 0x7fffffff;
        let value = ( seed / 0x7fffffff ) * 2 - 1; // Normalize to [-1, 1]

        // Apply method-specific transformations
        switch ( method )
        {
            case 'transformer':
                value *= Math.exp( -i / dimension ); // Exponential decay
                break;
            case 'bert':
                value *= Math.sin( i * Math.PI / dimension ); // Sinusoidal pattern
                break;
            case 'openai':
                value *= Math.tanh( value * 2 ); // Tanh activation
                break;
            case 'custom':
                value *= Math.cos( i * Math.PI / dimension ) * Math.exp( -i / ( dimension * 2 ) );
                break;
            default:
                // Standard normalization
                break;
        }

        embedding[ i ] = value;
    }

    // Normalize the embedding vector
    const magnitude = Math.sqrt( embedding.reduce( ( sum, val ) => sum + val * val, 0 ) );
    const normalizedEmbedding = embedding.map( val => val / magnitude );

    return {
        vector: normalizedEmbedding,
        magnitude: magnitude,
        method: method,
        dimension: dimension
    };
}

function computeCosineSimilarity ( vec1, vec2 )
{
    if ( vec1.length !== vec2.length )
    {
        throw new Error( 'Vectors must have the same dimension' );
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for ( let i = 0; i < vec1.length; i++ )
    {
        dotProduct += vec1[ i ] * vec2[ i ];
        norm1 += vec1[ i ] * vec1[ i ];
        norm2 += vec2[ i ] * vec2[ i ];
    }

    const magnitude = Math.sqrt( norm1 ) * Math.sqrt( norm2 );
    return magnitude === 0 ? 0 : dotProduct / magnitude;
}

function hashString ( str )
{
    let hash = 0;
    for ( let i = 0; i < str.length; i++ )
    {
        const char = str.charCodeAt( i );
        hash = ( ( hash << 5 ) - hash ) + char;
        hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs( hash ).toString( 16 );
}

function updateEmbeddingChart ( embedding )
{
    const canvas = document.getElementById( 'embeddingChart' );
    if ( !canvas ) return;

    const ctx = canvas.getContext( '2d' );

    // Clear previous chart
    if ( window.embeddingChartInstance )
    {
        window.embeddingChartInstance.destroy();
    }

    // Prepare data for visualization (first 50 dimensions)
    const dimensions = Math.min( 50, embedding.dimension );
    const labels = Array.from( { length: dimensions }, ( _, i ) => `D${ i + 1 }` );
    const data = embedding.vector.slice( 0, dimensions );

    window.embeddingChartInstance = new Chart( ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [ {
                label: 'Embedding Values',
                data: data,
                borderColor: 'rgba(79, 172, 254, 1)',
                backgroundColor: 'rgba(79, 172, 254, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            } ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Vector Embedding Visualization (${ embedding.dimension }D)`,
                    color: 'white'
                },
                legend: {
                    labels: {
                        color: 'white'
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: 'white' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    ticks: { color: 'white' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            }
        }
    } );
}

function updateTokenMetrics ()
{
    document.getElementById( 'total-supply' ).textContent =
        ( SystemState.metrics.totalSupply / 1000000 ).toFixed( 1 ) + 'M';
    document.getElementById( 'market-cap' ).textContent =
        '$' + ( SystemState.metrics.totalSupply * 0.5 / 1000000 ).toFixed( 1 ) + 'M';
}

function updateMetrics ()
{
    // Simulate real-time metrics updates
    SystemState.metrics.cpuUsage = Math.max( 20, Math.min( 80, SystemState.metrics.cpuUsage + ( Math.random() - 0.5 ) * 10 ) );
    SystemState.metrics.responseTime = Math.max( 100, Math.min( 2000, SystemState.metrics.responseTime + ( Math.random() - 0.5 ) * 200 ) );
    SystemState.metrics.successRate = Math.max( 95, Math.min( 100, SystemState.metrics.successRate + ( Math.random() - 0.5 ) * 2 ) );

    document.getElementById( 'cpu-usage' ).textContent = SystemState.metrics.cpuUsage.toFixed( 1 ) + '%';
    document.getElementById( 'network-speed' ).textContent = SystemState.metrics.responseTime.toFixed( 0 ) + 'ms';
    document.getElementById( 'success-rate' ).textContent = SystemState.metrics.successRate.toFixed( 1 ) + '%';

    updateTokenMetrics();
}

// ==================== TOAST NOTIFICATION SYSTEM ====================
function showToast ( message, type = 'info', duration = 4000 )
{
    const container = document.getElementById( 'toast-container' );
    const toast = document.createElement( 'div' );
    toast.className = `toast ${ type }`;

    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };

    toast.innerHTML = `${ icons[ type ] } ${ message }`;
    container.appendChild( toast );

    // Animate in
    setTimeout( () => toast.classList.add( 'show' ), 100 );

    // Remove after duration
    setTimeout( () =>
    {
        toast.classList.remove( 'show' );
        setTimeout( () =>
        {
            if ( container.contains( toast ) )
            {
                container.removeChild( toast );
            }
        }, 300 );
    }, duration );
}

function logOperation ( message, level = 'info' )
{
    const timestamp = new Date().toISOString();
    console.log( `[${ timestamp }] ${ level.toUpperCase() }: ${ message }` );

    // Store in session storage for debugging
    const logs = JSON.parse( sessionStorage.getItem( 'hmodel_logs' ) || '[]' );
    logs.push( { timestamp, level, message } );

    // Keep only last 1000 logs
    if ( logs.length > 1000 )
    {
        logs.splice( 0, logs.length - 1000 );
    }

    sessionStorage.setItem( 'hmodel_logs', JSON.stringify( logs ) );
}

// ==================== MODAL FUNCTIONS ====================
function openModal ( modalId )
{
    const modal = document.getElementById( modalId );
    if ( modal )
    {
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
    }
}

function closeModal ( modalId )
{
    const modal = document.getElementById( modalId );
    if ( modal )
    {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
}

function applySettings ()
{
    const apiEndpoint = document.getElementById( 'api-endpoint' ).value;
    const updateInterval = document.getElementById( 'update-interval' ).value;
    const debugMode = document.getElementById( 'debug-mode' ).value;
    const theme = document.getElementById( 'theme-selector' ).value;

    // Apply settings
    if ( apiEndpoint )
    {
        sessionStorage.setItem( 'hmodel_api_endpoint', apiEndpoint );
    }

    sessionStorage.setItem( 'hmodel_update_interval', updateInterval );
    sessionStorage.setItem( 'hmodel_debug_mode', debugMode );
    sessionStorage.setItem( 'hmodel_theme', theme );

    // Apply theme if different from default
    if ( theme !== 'default' )
    {
        applyTheme( theme );
    }

    showToast( 'Settings applied successfully', 'success' );
    closeModal( 'settings-modal' );

    logOperation( `Settings applied: theme=${ theme }, debug=${ debugMode }, interval=${ updateInterval }s`, 'info' );
}

function resetSettings ()
{
    // Reset form values
    document.getElementById( 'api-endpoint' ).value = 'https://api.hmodel.ai/v1';
    document.getElementById( 'update-interval' ).value = '30';
    document.getElementById( 'debug-mode' ).value = 'off';
    document.getElementById( 'theme-selector' ).value = 'default';

    // Clear stored settings
    sessionStorage.removeItem( 'hmodel_api_endpoint' );
    sessionStorage.removeItem( 'hmodel_update_interval' );
    sessionStorage.removeItem( 'hmodel_debug_mode' );
    sessionStorage.removeItem( 'hmodel_theme' );

    showToast( 'Settings reset to defaults', 'info' );
}

function exportSettings ()
{
    const settings = {
        apiEndpoint: document.getElementById( 'api-endpoint' ).value,
        updateInterval: document.getElementById( 'update-interval' ).value,
        debugMode: document.getElementById( 'debug-mode' ).value,
        theme: document.getElementById( 'theme-selector' ).value,
        exportedAt: new Date().toISOString()
    };

    const blob = new Blob( [ JSON.stringify( settings, null, 2 ) ], { type: 'application/json' } );
    const url = URL.createObjectURL( blob );

    const a = document.createElement( 'a' );
    a.href = url;
    a.download = `hmodel_settings_${ new Date().toISOString().split( 'T' )[ 0 ] }.json`;
    a.click();

    URL.revokeObjectURL( url );
    showToast( 'Settings exported successfully', 'success' );
}

function applyTheme ( theme )
{
    // This would implement different color schemes
    const root = document.documentElement;

    switch ( theme )
    {
        case 'dark':
            root.style.setProperty( '--primary-gradient', 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)' );
            root.style.setProperty( '--secondary-gradient', 'linear-gradient(135deg, #16213e 0%, #1a1a2e 100%)' );
            root.style.setProperty( '--accent-color', '#4f9cef' );
            root.style.setProperty( '--text-primary', '#ffffff' );
            root.style.setProperty( '--text-secondary', '#b3b3b3' );
            root.style.setProperty( '--background-primary', '#1a1a2e' );
            root.style.setProperty( '--background-secondary', '#16213e' );
            root.style.setProperty( '--border-color', 'rgba(255,255,255,0.1)' );
            root.style.setProperty( '--shadow-color', 'rgba(255,255,255,0.05)' );
            root.style.setProperty( '--chart-background', 'rgba(255,255,255,0.05)' );
            root.style.setProperty( '--chart-grid', 'rgba(255,255,255,0.03)' );
            root.style.setProperty( '--chart-axis', 'rgba(255,255,255,0.1)' );
            root.style.setProperty( '--chart-text', 'rgba(255,255,255,0.8)' );
            root.style.setProperty( '--chart-line', 'rgba(79,172,254,0.8)' );
            root.style.setProperty( '--chart-area', 'rgba(79,172,254,0.1)' );
            root.style.setProperty( '--chart-point', 'rgba(79,172,254,1)' );
            root.style.setProperty( '--chart-area-fill', 'rgba(79,172,254,0.05)' );
            root.style.setProperty( '--chart-area-stroke', 'rgba(79,172,254,0.2)' );
            root.style.setProperty( '--toast-success', 'rgba(79,172,254,0.1)' );
            root.style.setProperty( '--toast-error', 'rgba(255,99,71,0.1)' );
            root.style.setProperty( '--toast-warning', 'rgba(255,215,0,0.1)' );
            root.style.setProperty( '--toast-info', 'rgba(128,128,128,0.1)' );
            root.style.setProperty( '--toast-text', '#ffffff' );
            root.style.setProperty( '--toast-border', 'rgba(255,255,255,0.1)' );
            root.style.setProperty( '--toast-shadow', 'rgba(255,255,255,0.05)' );
            root.style.setProperty( '--toast-icon-success', '✅' );
            root.style.setProperty( '--toast-icon-error', '❌' );
            root.style.setProperty( '--toast-icon-warning', '⚠️' );
            root.style.setProperty( '--toast-icon-info', 'ℹ️' );
            break;
        case 'light':
            root.style.setProperty( '--primary-gradient', 'linear-gradient(135deg, #f0f0f0 0%, #d9d9d9 100%)' );
            root.style.setProperty( '--secondary-gradient', 'linear-gradient(135deg, #d9d9d9 0%, #f0f0f0 100%)' );
            root.style.setProperty( '--accent-color', '#4f9cef' );
            root.style.setProperty( '--text-primary', '#333333' );
            root.style.setProperty( '--text-secondary', '#666666' );
            root.style.setProperty( '--background-primary', '#f0f0f0' );
            root.style.setProperty( '--background-secondary', '#d9d9d9' );
            root.style.setProperty( '--border-color', 'rgba(0,0,0,0.1)' );
            root.style.setProperty( '--shadow-color', 'rgba(0,0,0,0.05)' );
            root.style.setProperty( '--chart-background', 'rgba(255,255,255,0.95)' );
            root.style.setProperty( '--chart-grid', 'rgba(0,0,0,0.03)' );
            root.style.setProperty( '--chart-axis', 'rgba(0,0,0,0.1)' );
            root.style.setProperty( '--chart-text', 'rgba(0,0,0,0.8)' );
            root.style.setProperty( '--chart-line', 'rgba(79,172,254,0.8)' );
            root.style.setProperty( '--chart-area', 'rgba(79,172,254,0.1)' );
            root.style.setProperty( '--chart-point', 'rgba(79,172,254,1)' );
            root.style.setProperty( '--chart-area-fill', 'rgba(79,172,254,0.05)' );
            root.style.setProperty( '--chart-area-stroke', 'rgba(79,172,254,0.2)' );
            root.style.setProperty( '--toast-success', 'rgba(79,172,254,0.1)' );
            root.style.setProperty( '--toast-error', 'rgba(255,99,71,0.1)' );
            root.style.setProperty( '--toast-warning', 'rgba(255,215,0,0.1)' );
            root.style.setProperty( '--toast-info', 'rgba(128,128,128,0.1)' );
            root.style.setProperty( '--toast-text', '#333333' );
            root.style.setProperty( '--toast-border', 'rgba(0,0,0,0.1)' );
            root.style.setProperty( '--toast-shadow', 'rgba(0,0,0,0.05)' );
            root.style.setProperty( '--toast-icon-success', '✅' );
            root.style.setProperty( '--toast-icon-error', '❌' );
            root.style.setProperty( '--toast-icon-warning', '⚠️' );
            root.style.setProperty( '--toast-icon-info', 'ℹ️' );
            break;
        case 'neon':
            root.style.setProperty( '--primary-gradient', 'linear-gradient(135deg, #00ff00 0%, #008000 100%)' );
            root.style.setProperty( '--secondary-gradient', 'linear-gradient(135deg, #008000 0%, #00ff00 100%)' );
            root.style.setProperty( '--accent-color', '#00ff00' );
            root.style.setProperty( '--text-primary', '#00ff00' );