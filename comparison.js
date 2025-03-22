function calculateAngle(a, b, c) {
    const p1 = [a.x, a.y];
    const p2 = [b.x, b.y];
    const p3 = [c.x, c.y];
    
    const radians = Math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - 
                   Math.atan2(p1[1] - p2[1], p1[0] - p2[0]);
    let angle = Math.abs(radians * 180.0 / Math.PI);
    
    if (angle > 180.0) {
        angle = 360 - angle;
    }
    return angle;
}

function extractAngles(landmarks) {
    if (!landmarks) return null;
    
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    const leftHip = landmarks[23];
    const rightHip = landmarks[24];
    const leftKnee = landmarks[25];
    const rightKnee = landmarks[26];
    const leftAnkle = landmarks[27];
    const rightAnkle = landmarks[28];
    
    const angles = {
        shoulders: {
            left: calculateAngle(leftHip, leftShoulder, 
                   {x: leftShoulder.x + 0.1, y: leftShoulder.y}),
            right: calculateAngle(rightHip, rightShoulder, 
                    {x: rightShoulder.x + 0.1, y: rightShoulder.y})
        },
        hips: {
            left: calculateAngle(leftShoulder, leftHip, leftKnee),
            right: calculateAngle(rightShoulder, rightHip, rightKnee)
        },
        knees: {
            left: calculateAngle(leftHip, leftKnee, leftAnkle),
            right: calculateAngle(rightHip, rightKnee, rightAnkle)
        },
        ankles: {
            left: calculateAngle(leftKnee, leftAnkle, 
                   {x: leftAnkle.x, y: leftAnkle.y + 0.1}),
            right: calculateAngle(rightKnee, rightAnkle, 
                    {x: rightAnkle.x, y: rightAnkle.y + 0.1})
        }
    };
    
    return angles;
}

async function compareWithModel(userFrames) {
    try {
        console.log("Starting comparison...");
        console.log(`User frames received: ${userFrames.length}`);
        
        const modelResponse = await fetch('model_landmarks.json');
        if (!modelResponse.ok) {
            throw new Error('Failed to load model data');
        }
        const modelFrames = await modelResponse.json();
        console.log(`Model frames loaded: ${modelFrames.length}`);
        
        const angleDifferences = {
            shoulders: { left: [], right: [] },
            hips: { left: [], right: [] },
            knees: { left: [], right: [] },
            ankles: { left: [], right: [] }
        };
        
        // Sample frames at regular intervals to match model and user frames
        const frameCount = Math.min(userFrames.length, modelFrames.length);
        console.log(`Comparing ${frameCount} frames`);
        
        for (let i = 0; i < frameCount; i++) {
            const userLandmarks = userFrames[i].landmarks;
            const modelLandmarks = modelFrames[i];
            
            const userAngles = extractAngles(userLandmarks);
            const modelAngles = extractAngles(modelLandmarks);
            
            if (userAngles && modelAngles) {
                for (const joint in angleDifferences) {
                    for (const side of ['left', 'right']) {
                        const diff = Math.abs(modelAngles[joint][side] - userAngles[joint][side]);
                        angleDifferences[joint][side].push(diff);
                        console.log(`${joint} ${side} difference: ${diff.toFixed(2)}`);
                    }
                }
            }
        }
        
        const scores = {};
        let totalScore = 0;
        
        for (const joint in angleDifferences) {
            const leftDiffs = angleDifferences[joint].left;
            const rightDiffs = angleDifferences[joint].right;
            
            if (leftDiffs.length > 0 || rightDiffs.length > 0) {
                const leftAvg = leftDiffs.length > 0 ? 
                    leftDiffs.reduce((a, b) => a + b) / leftDiffs.length : Infinity;
                const rightAvg = rightDiffs.length > 0 ? 
                    rightDiffs.reduce((a, b) => a + b) / rightDiffs.length : Infinity;
                
                const bestDiff = Math.min(leftAvg, rightAvg);
                const score = Math.max(0, 25 - (bestDiff / 45) * 25);
                scores[joint] = score;
                totalScore += score;
                
                console.log(`${joint} score: ${score.toFixed(2)}/25 (diff: ${bestDiff.toFixed(2)})`);
            } else {
                scores[joint] = 0;
                console.log(`${joint}: No valid measurements`);
            }
        }
        
        console.log(`Total score: ${totalScore.toFixed(2)}/100`);
        
        return {
            scores: scores,
            totalScore: totalScore,
            framesAnalyzed: frameCount,
            angleDifferences: angleDifferences
        };
    } catch (error) {
        console.error('Comparison error:', error);
        throw error;
    }
} 